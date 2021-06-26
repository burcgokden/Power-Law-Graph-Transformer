
import tensorflow as tf
import functools as ft

class plga_layer(tf.keras.layers.Layer):
    '''
    Power law graph attention layer implementation.
    '''
    def __init__(self, F_hidden, att_head=1, in_dropout_prob=0.0, eij_dropout_prob=0.0, activation=tf.nn.relu,
                 att_activation = tf.nn.leaky_relu, pooling=None, a_regularizer=None,
                 W_regularizer=None, pw_regularizer=None, a_init=None,
                 W_init=None, b_init=None, pw_init=None, **kwargs):
        '''
        Args:
            F_hidden: hidden layer shape used in layer weight creation. For multi-head plga this is depth.
            att_head: additive head number. For multi-head plga this is 1.
            in_dropout_prob: drop out rate for query, key inputs
            eij_dropout_prob: drop out rate for attention weight
            activation: activation applied to attention output.
            att_activation: activation used on final attention model output.
            pooling: If true, the additive heads are averaged, otherwise they are concatenated.
            W_regularizer: regularizer for weight used in attention model
            pw_regularizer: regularizer for learnable power weights.
            a_init: initializer for learnable coupling coefficients.
            W_init: initializer for weight used in attention model
            b_init: initializer for bias values.
            pw_init: initializer for power weights.
        '''

        super().__init__(**kwargs)
        self.F_hidden=F_hidden
        self.att_head=att_head
        self.in_dropout_prob = in_dropout_prob
        self.eij_dropout_prob = eij_dropout_prob
        self.W_regularizer = W_regularizer
        self.a_regularizer = a_regularizer
        self.pw_regularizer = pw_regularizer
        self.W_initializer = W_init if W_init is not None else tf.keras.initializers.glorot_normal
        self.pw_initializer = pw_init if pw_init is not None else  tf.keras.initializers.glorot_normal
        self.a_initializer=a_init if a_init is not None else tf.keras.initializers.glorot_normal
        self.b_initializer=b_init if b_init is not None else tf.keras.initializers.Zeros
        self.pooling=pooling
        self.activation = tf.keras.activations.get(activation)
        self.att_activation = tf.keras.activations.get(att_activation)

        self.dropout1 = tf.keras.layers.Dropout(rate=self.in_dropout_prob)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.in_dropout_prob)
        self.dropout3 = tf.keras.layers.Dropout(rate=self.eij_dropout_prob)

    #make this just the alignment model, returning single attention coefficients
    def cg_align_one(self, Hin, Hkt, A, a_vec, ba, W, b, pw, mask=None):
        '''
        alignment model for calculating E with elements eij
        Args:
            Hin: query
            Hkt: transpose of key
            A: metric tensor instance
            a_vec: learnable coupling coefficients.
            ba: bias for coupling coeffients
            W: weights appliead on metric tensor before ReLU
            b: bias applied on metric tensor before ReLU
            pw:power values applied on metric tensor
            mask: padding or lookahead mask
        Returns:
            E: attention weights applied on value
            AW: metric tensor after activation is applied
            pw: learned power values
            a_vec: learned coupling coefficients
            ba: bias for coupling coefficients
            avAp: Energy curvature tensor
            Ep: Energy-curvature tensor before mask is applied
        '''

        We = tf.tile(W[tf.newaxis, :,:,:], tf.stack([tf.shape(Hin)[0], 1, 1, 1]))  # [batch_size, num_head, depth, depth]
        a_vece = tf.tile(a_vec[tf.newaxis, :,:,:], tf.stack([tf.shape(Hin)[0], 1, 1, 1]))  # [batch_size, num_head, depth, depth]
        AdjActivation=tf.nn.relu
        epsilonAdj=1e-9

        #make metric tensor positive definite
        AW=AdjActivation(tf.matmul(We,A)+b)+epsilonAdj

        #find energy curvature tensor and attention weights
        pwe = tf.tile(pw[tf.newaxis, :,:,:], tf.stack([tf.shape(Hin)[0], 1, 1, 1]))  # [batch_size, num_head,  depth, depth]
        Ap=tf.math.pow(AW,pwe, name="Adj_pow")
        avAp=tf.matmul(a_vece, Ap)+ba # [batch_size, num_head,  depth, depth]
        WHiWHj = tf.matmul(Hin, avAp) #[batch_size, num_head, seq_lenq, depth]
        Ep=tf.matmul(WHiWHj, Hkt) #[batch_size, num_head, seq_lenq, seq_lenk]

        #scale attention with square root of depth
        dk=tf.cast(self.F_hidden, tf.float32)
        Ep=Ep/tf.math.sqrt(dk)
        Ep=self.att_activation(Ep)

        #apply mask and softmask
        E= Ep + (mask * -1e9) if mask is not None else Ep
        E=tf.nn.softmax(E, axis=-1)

        return E, AW, pw, a_vec, ba, avAp, Ep


    def cg_single_head(self, Hin, Hkt, Hv, A, a_vec, ba, W, b, pw, mask=None, training=None):
        '''
        Method for linear propagation of attention weights over values.
        '''

        Eout, AW_out, pw_out, avec_out, ba_out, avAp_out, Ep_out=self.cg_align_one(Hin, Hkt, A, a_vec, ba, W, b, pw, mask=mask)

        Eout=self.dropout3(Eout, training=training)

        Hout = tf.matmul(Eout, Hv) #[batch_size, num_heads, seq_lenq ,d_model]

        return Hout, Eout, AW_out, pw_out, avec_out, ba_out, avAp_out, Ep_out

    def cg_multi_head(self, Hin, Hk, Hv, A, mask=None, training=None): #(self, Hin, A, Hk, Hv, mask=None, training=None): #
        '''
        Method for additive attention head and multi-head calculation.
        '''
        Hlst=[]
        Elst=[]
        AWlst=[]
        pwlst=[]
        alst_out=[]
        balst_out=[]
        avAplst_out =[]
        Eplst_out = []

        Hin = self.dropout1(Hin, training=training)
        Hk =  self.dropout2(Hk, training=training)

        Hkt = tf.transpose(Hk, perm=[0, 1, 3, 2])  # (batch_size, num_head, depth, seq_lenk)

        for i in range(self.att_head):
            Hout, Eout, AW_out, pw_out, avec_out, ba_out, avAp_out, Ep_out=self.cg_single_head(Hin, Hkt, Hv, A,
                                                          self.alst[i],
                                                          self.balst[i],
                                                          self.Wlst[i],
                                                          self.blst[i],
                                                          self.pwlst[i],
                                                          mask=mask,
                                                          training=training)
            Hout=self.activation(Hout)
            Hlst.append(Hout)
            Elst.append(Eout)
            AWlst.append(AW_out)
            pwlst.append(pw_out)
            alst_out.append(avec_out)
            balst_out.append(ba_out)
            avAplst_out.append(avAp_out)
            Eplst_out.append(Ep_out)

        if self.pooling:
            H_next=tf.add_n(Hlst)/self.att_head
        else:
            H_next = tf.concat(Hlst, axis=-1)
        return H_next, Elst, AWlst, pwlst, alst_out, balst_out, avAplst_out, Eplst_out


    def build(self, input_shape):
        '''
        Used to initialize learnable parameters for the layer:
        W: weights to apply on metric tensor
        b: bias to apply on metric tensor
        a: coupling coefficients for power law attention
        ba: bias for power law attention.
        pw: power weights for power law attention
        '''

        X_shape=input_shape[0][1:] #[num_heads, seq-len, depth]

        add_weight_Wpart=ft.partial(self.add_weight, shape=(X_shape[0], self.F_hidden, X_shape[2]),
                                   trainable=True,
                                 regularizer=self.W_regularizer,
                                 initializer=self.W_initializer)
        add_weight_bpart=ft.partial(self.add_weight, shape=(X_shape[0], self.F_hidden, X_shape[2]),
                                   trainable=True,
                                 regularizer=None,
                                 initializer=self.b_initializer)
        add_weight_pwpart=ft.partial(self.add_weight, shape=(X_shape[0], X_shape[2], X_shape[2]),
                                   trainable=True,
                                 regularizer=self.pw_regularizer,
                                 initializer=self.pw_initializer)
        add_weight_apart = ft.partial(self.add_weight, shape=(X_shape[0], self.F_hidden, X_shape[2]),
                                       trainable=True,
                                       regularizer=self.a_regularizer,
                                       initializer=self.a_initializer)
        add_weight_bapart=ft.partial(self.add_weight, shape=(X_shape[0], self.F_hidden, X_shape[2]),
                                   trainable=True,
                                 regularizer=None,
                                 initializer=self.b_initializer)

        #create a list of weights for W,a,b and pw as  much as number of additive attention heads
        self.Wlst = [add_weight_Wpart(name="weight"+str(i)) for i in range(self.att_head)]
        self.blst = [add_weight_bpart(name="bias_w"+str(i)) for i in range(self.att_head)]
        self.balst = [add_weight_bapart(name="bias_a" + str(i)) for i in range(self.att_head)]
        self.pwlst = [add_weight_pwpart(name="adj_power"+str(i)) for i in range(self.att_head)]
        self.alst = [add_weight_apart(name="att_vector"+str(i)) for i in range(self.att_head)]

        super().build(input_shape)


    def call(self, inputs, training=None, **kwargs):
        '''
        execute the forward propagation
        inputs[0]=query = Hin
        inputs[1]=key =Hk
        inputs[2]=value = Hv
        inputs[3]=metric tensor = A
        inputs[4]=mask
        '''
        Hin, Hk, Hv, A, mask=inputs
        H_next, Elst, AWlst, pwlst, alst_out, balst_out, avAplst_out, Eplst_out = self.cg_multi_head(Hin, Hk, Hv, A, mask=mask, training=training) #
        return [H_next, Elst, AWlst, pwlst, alst_out, balst_out, avAplst_out, Eplst_out]

    def get_config(self):
        config = super().get_config()
        config= config.update({"F_hidden": self.F_hidden,
                               "att_head": self.att_head,
                               "in_keep_prob": self.in_keep_prob,
                               "eij_keep_prob": self.eij_keep_prob,
                               "activation": self.activation,
                               "att_activation": self.att_activation,
                               "pooling": self.pooling,
                               "W_regularizer": self.W_regularizer,
                               "a_regularizer": self.a_regularizer,
                               "pw_regularizer": self.pw_regularizer,
                               "W_init": self.W_initializer,
                               "b_init": self.b_initializer,
                               "pw_init": self.pw_initializer,
                               "a_init": self.a_initializer,
                               "post_att_fnn":self.post_att_ffn
        })
        return config
