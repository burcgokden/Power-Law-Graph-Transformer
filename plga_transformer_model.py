
import numpy as np

import tensorflow as tf
import power_law_attention_layer as plgatt


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, att_dropout_rate_in=0.0,
                 att_dropout_rate_eij=0.0, Adropout_rate=0.0, A_dff=None,
                 num_reslayerA=None, num_denseA=None, **kwargs):

        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.att_dropout_rate_in=att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff = A_dff if A_dff is not None else self.d_model
        self.num_denseA = num_denseA if num_denseA is not None else 1
        self.num_reslayerA = num_reslayerA if num_reslayerA is not None else 1

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')

        self.plgatt_layer= plgatt.plga_layer(F_hidden=self.depth, att_head=1,
                                       activation=None,
                                       pw_regularizer=None,
                                       in_dropout_prob=self.att_dropout_rate_in,
                                       eij_dropout_prob=self.att_dropout_rate_eij,
                                       name='plga_layer')

        self.dense = tf.keras.layers.Dense(d_model, name='dense')

        #residual layers for metric tensor learning
        self.reslayerAs=[ResLayerA(depth=self.depth, A_dff=self.A_dff,
                                  Adropout_rate=self.Adropout_rate,
                                  num_denseA=self.num_denseA,
                                  index=str(i)) for i in range(self.num_reslayerA)]

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None, **kwargs):
        '''
        Args:
            inputs: [q,k,v,mask]
            training
        Returns:
            inductive and deductive task outputs.
        '''
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)


        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        #Calculate density matrix using linear self attention
        qt = tf.transpose(q, perm=[0,1, 3, 2])
        A = tf.matmul(qt, q)  # (batch_size, num_head, depth, depth)

        #Deep residual network for learning metric tensor
        for i in range(self.num_reslayerA):
            A=self.reslayerAs[i]([A], training=training)

        #Apply multi-head power law attention
        Hnext, Elst, Alst, pwlst, attvlst, balst, avAplst, Eplst = self.plgatt_layer([q, k, v, A, mask], training=training)
        Hnext = tf.transpose(Hnext, perm=[0, 2, 1, 3])

        Hnext= tf.reshape(Hnext, (batch_size, -1, self.d_model)) # [batch_size, seq_len, d_model]

        output = self.dense(Hnext)

        return output, Elst, Alst, pwlst, attvlst, balst, avAplst, Eplst

    def get_config(self):
        config = super().get_config()
        config=config.update({
                    "d_model":self.d_model,
                    "num_heads":self.num_heads,
                    "wq":self.wq,
                    "wk":self.wk,
                    "wv":self.wv,
                    "cgattL":self.cgattL,
                    "dense":self.dense,
                    "Afnn":self.Afnn,
                    "att_dropout_rate": self.att_droput_rate,
                    "Adropout_rate": self.Adropout_rate,
                    "A_dff": self.A_dff,
                    "num_denseA": self.num_denseA,
                    "num_reslayerA": self.num_reslayerA
                    })
        return config


class EncoderLayer(tf.keras.layers.Layer):
    '''
    Single encoder layer implementation
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff=None, num_reslayerA=None, num_denseA=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff = A_dff
        self.num_denseA=num_denseA
        self.num_reslayerA = num_reslayerA

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.att_dropout_rate_in,
                                      self.att_dropout_rate_eij, self.Adropout_rate, self.A_dff,
                                      self.num_reslayerA, self.num_denseA)
        self.ffn = self.enc_point_wise_feed_forward_network()

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm2')

        self.dropout1 = tf.keras.layers.Dropout(self.rate, name='dropout1')
        self.dropout2 = tf.keras.layers.Dropout(self.rate, name='dropout2')

    def call(self, inputs, training=None, **kwargs):
        '''
        inputs: [x, mask].
        Returns encoder output and deductive task outputs for SLM attention block.
        '''
        x, mask = inputs
        attn_output, Elst, Alst, pwlst, attvlst, balst, avAplst, Eplst = self.mha([x, x, x, mask], training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, [Elst, Alst, pwlst, attvlst, balst, avAplst, Eplst]


    def enc_point_wise_feed_forward_network(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu', name='dense1'),
            tf.keras.layers.Dense(self.d_model, name='dense2')
        ])

    def get_config(self):
        config = super().get_config()
        config=config.update({
                    "d_model":self.d_model,
                    "num_heads":self.num_heads,
                    "dff":self.dff,
                    "rate":self.rate,
                    "mha":self.mha,
                    "layernorm1":self.layernorm1,
                    "layernorm2":self.layernorm2,
                    "dropout1":self.dropout1,
                    "dropout2":self.dropout2,
                    "att_dropout_rate_in":self.att_droput_rate_in,
                    "att_dropout_rate_eij": self.att_dropout_rate_eij,
                    "Adropout_rate": self.Adropout_rate,
                    "A_dff":self.A_dff,
                    "num_denseA": self.num_denseA,
                    "num_reslayerA": self.num_reslayerA
                    })
        return config



class DecoderLayer(tf.keras.layers.Layer):
    '''
    Single decoder layer implementation.
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff=None, num_reslayerA=None, num_denseA=None, **kwargs):

        super().__init__(**kwargs)

        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.rate=rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff=A_dff
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA

        self.mha1 = MultiHeadAttention(self.d_model, self.num_heads, self.att_dropout_rate_in,
                                       self.att_dropout_rate_eij, self.Adropout_rate, self.A_dff,
                                       self.num_reslayerA, self.num_denseA)
        self.mha2 = MultiHeadAttention(self.d_model, self.num_heads, self.att_dropout_rate_in,
                                       self.att_dropout_rate_eij, self.Adropout_rate, self.A_dff,
                                       self.num_reslayerA, self.num_denseA)

        self.ffn = self.dec_point_wise_feed_forward_network()

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm2')
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm3')

        self.dropout1 = tf.keras.layers.Dropout(self.rate, name='dropout1')
        self.dropout2 = tf.keras.layers.Dropout(self.rate, name='dropout2')
        self.dropout3 = tf.keras.layers.Dropout(self.rate, name='dropout3')

    def call(self, inputs, training=None, **kwargs):
        '''
        inputs: [x, enc_output, look_ahead_mask, padding_mask ]
        Returns decoder output and deductive task outputs for TLM and XLM attention blocks.
        '''
        x, enc_output, look_ahead_mask, padding_mask = inputs

        attn1, Elst1, Alst1, pwlst1, attvlst1, balst1, avAplst1, Eplst1 = self.mha1([x,x,x, look_ahead_mask], training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, Elst2, Alst2, pwlst2, attvlst2, balst2, avAplst2, Eplst2 = self.mha2([out1, enc_output, enc_output, padding_mask], training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, [Elst1, Alst1, pwlst1, attvlst1, balst1, avAplst1, Eplst1], [Elst2, Alst2, pwlst2, attvlst2, balst2, avAplst2, Eplst2]

    def dec_point_wise_feed_forward_network(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'),
            tf.keras.layers.Dense(self.d_model)
        ])

    def get_config(self):
        config = super().get_config()
        config = config.update({
                  "d_model":self.d_model,
                  "num_heads":self.num_heads,
                  "dff":self.dff,
                  "rate":self.rate,
                  "mha1":self.mha1,
                  "mha2":self.mha2,
                  "ffn":self.ffn,
                  "layernorm1":self.layernorm1,
                  "layernorm2":self.layernorm2,
                  "layernorm3":self.layernorm3,
                  "dropou1": self.dropout1,
                  "dropout2": self.dropout2,
                  "dropout3": self.dropout3,
                  "att_dropout_rate_in": self.att_droput_rate_in,
                  "att_dropout_rate_eij": self.att_dropout_rate_eij,
                  "A_dff": self.A_dff,
                  "num_denseA": self.num_denseA,
                  "num_reslayerA": self.num_reslayerA
                  })
        return config


class Encoder(tf.keras.layers.Layer):
    '''
    Multilayer encoder implementation.
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0,  A_dff=None, num_reslayerA=None, num_denseA=None, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.dff=dff
        self.input_vocab_size=input_vocab_size
        self.maximum_position_encoding=maximum_position_encoding
        self.rate =rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff=A_dff
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA

        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model, name='enc_embedding')
        self.pos_encoding = self.positional_encoding(self.maximum_position_encoding)

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate, self.att_dropout_rate_in,
                                        self.att_dropout_rate_eij, self.Adropout_rate, self.A_dff,
                                        self.num_reslayerA, self.num_denseA) for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)


    def call(self, inputs, training=None, **kwargs):
        '''
        inputs: [x, mask].
        Returns output of encoder and attention weights for SLM attention block.
        '''
        x, mask = inputs
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        enc_att_weights  = []
        for i in range(self.num_layers):
            x, enc_att_w = self.enc_layers[i]([x, mask],training=training)
            enc_att_weights.append(enc_att_w)

        return x, enc_att_weights

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def positional_encoding(self, position):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(self.d_model)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_config(self):
        config=super().get_config()
        config = config.update({
                  "num_layers":self.num_layers,
                  "d_model":self.d_model,
                  "num_heads":self.num_heads,
                  "dff":self.dff,
                  "input_vocab_size":self.input_vocab_size,
                  "maximum_position_encoding":self.maximum_position_encoding,
                  "rate":self.rate,
                  "embedding":self.embedding,
                  "pos_encoding":self.pos_encoding,
                  "enc_layers": self.enc_layers,
                  "dropout":self.dropout,
                  "att_dropout_rate_in": self.att_droput_rate_in,
                  "att_dropout_rate_eij": self.att_dropout_rate_eij,
                  "Adropout_rate": self.Adropout_rate,
                  "A_dff": self.A_dff,
                  "num_denseA": self.num_denseA,
                  "num_reslayerA": self.num_reslayerA
                  })
        return config


class Decoder(tf.keras.layers.Layer):
    '''
    Multi layer decoder implementation
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff=None, num_reslayerA=None, num_denseA=None, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.dff=dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff=A_dff
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA

        self.embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model, name='dec_embedding')
        self.pos_encoding = self.positional_encoding(self.maximum_position_encoding)

        self.dec_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate, self.att_dropout_rate_in,
                                        self.att_dropout_rate_eij, self.Adropout_rate, self.A_dff,
                                        self.num_reslayerA, self.num_denseA) for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate, name='dropout')


    def call(self, inputs, training=None, **kwargs):
        '''
        inputs: [x, enc_output, look_ahead_mask, padding_mask].
        Returns output of decoder and attention weights for TLM and XLM attention blocks.
        '''
        x, enc_output, look_ahead_mask, padding_mask = inputs
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        dec_att_weigths1, dec_att_weights2=[],[]
        for i in range(self.num_layers):
            x, dec_att_w1, dec_att_w2 = self.dec_layers[i]([x, enc_output, look_ahead_mask, padding_mask], training=training)
            dec_att_weigths1.append(dec_att_w1)
            dec_att_weights2.append(dec_att_w2)

        return x, dec_att_weigths1, dec_att_weights2

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def positional_encoding(self, position):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(self.d_model)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


    def get_config(self):
        config=super().get_config()
        config = config.update({
                  "num_layers":self.num_layers,
                  "d_model":self.d_model,
                  "num_heads":self.num_heads,
                  "dff":self.dff,
                  "target_vocab_size":self.target_vocab_size,
                  "maximum_position_encoding":self.maximum_position_encoding,
                  "rate":self.rate,
                  "embedding":self.embedding,
                  "pos_encoding":self.pos_encoding,
                  "dec_layers":self.dec_layers,
                  "dropout":self.dropout,
                  "att_dropout_rate_in":self.att_droput_rate_in,
                  "att_dropout_rate_eij": self.att_dropout_rate_eij,
                  "Adropout_rate": self.Adropout_rate,
                  "A_dff": self.A_dff,
                  "num_denseA": self.num_denseA,
                  "num_reslayerA": self.num_reslayerA
                  })
        return config


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 pe_input, pe_target, rate=0.1, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff=None, num_reslayerA=None, num_denseA=None, **kwargs):
        '''
        Args:
            num_layers: number of encoder-decoder layers.
            d_model: embedding/LM feature dimension
            num_heads: number of attention heads
            dff: number of neurons on single layer of fully connected network
            input_vocab_size: vocabulary size for source language
            target_vocab_size: vocabulary size for target language
            pe_input: maximum positional encoding number for source
            pe_target: maximum positional encoding number for target
            rate: drop out rate for embeddings and output of fully connected networks.
            att_dropout_rate_in: drop out rate for power law attention query and key inputs
            att_dropout_rate_eij: drop out rate for power law attention weight
            Adropout_rate: drop out rate for each unit in residual network for metric tensor learning
            A_dff:;Number of neurons in single layer of residual unit for metric tensor learning
            num_reslayerA: number of residual units
            num_denseA: number of dense layers in each residual unit.
        Returns:
            Logit probabilities for predicted sentence and power law attention weights for deductive task.

        '''
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pe_input=pe_input
        self.pe_target= pe_target
        self.rate=rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff = A_dff
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA

        self.tokenizer = Encoder(self.num_layers, self.d_model, self.num_heads, self.dff,
                                 self.input_vocab_size, self.pe_input, self.rate, self.att_dropout_rate_in,
                                 self.att_dropout_rate_eij, self.Adropout_rate, self.A_dff,
                                 self.num_reslayerA, self.num_denseA)

        self.decoder = Decoder(self.num_layers, self.d_model, self.num_heads, self.dff,
                               self.target_vocab_size, self.pe_target, self.rate, self.att_dropout_rate_in,
                               self.att_dropout_rate_eij,  self.Adropout_rate, self.A_dff,
                               self.num_reslayerA, self.num_denseA)

        self.final_layer = tf.keras.layers.Dense(self.target_vocab_size,name='dense_final_layer')

    def call(self, inputs, training=None, **kwargs):
        inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask=inputs
        enc_output, enc_att_weights = self.tokenizer([inp, enc_padding_mask], training=training)

        dec_output, dec_att_weights1, dec_att_weights2 = self.decoder([tar, enc_output, look_ahead_mask, dec_padding_mask], training=training )

        final_output = self.final_layer(dec_output)

        return final_output, [enc_att_weights, dec_att_weights1, dec_att_weights2]


    def get_config(self):
        config=super().get_config()
        config = config.update({
                  "num_layers":self.num_layers,
                  "d_model":self.d_model,
                  "num_heads":self.num_heads,
                  "dff":self.dff,
                  "input_vocab_size":self.input_vocab_size,
                  "target_vocab_size":self.target_vocab_size,
                  "pe_input":self.pe_input,
                  "pe_target":self.pe_target,
                  "rate":self.rate,
                  "final_layer":self.final_layer,
                  "tokenizer": self.tokenizer,
                  "decoder": self.decoder,
                  "att_dropout_rate_in":self.att_droput_rate_in,
                  "att_dropout_rate_eij": self.att_dropout_rate_eij,
                  "Adropout_rate": self.Adropout_rate,
                  "A_dff": self.A_dff,
                  "num_denseA":self.num_denseA,
                  "num_reslayerA":self.num_reslayerA
                  })
        return config


class ResLayerA(tf.keras.layers.Layer):
    def __init__(self, depth, A_dff, Adropout_rate=0.0, num_denseA=None, index='0', **kwargs):
        super().__init__(**kwargs)
        self.depth=depth
        self.A_dff = A_dff
        self.Adropout_rate = Adropout_rate
        self.num_denseA = num_denseA if num_denseA is not None else 1
        self.index=index

        self.denseAs = [tf.keras.layers.Dense(self.A_dff, activation='relu', name="denseA"+self.index+str(i))
                        for i in range(self.num_denseA)]
        self.dropoutA = tf.keras.layers.Dropout(rate=self.Adropout_rate, name="Adropout"+self.index)
        self.denseA = tf.keras.layers.Dense(self.depth, name="denseA"+self.index)
        self.layernormA = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernormA'+self.index)

    def ResUnit(self, A, training):
        Ain = tf.identity(A)
        for i in range(self.num_denseA):
            A = self.denseAs[i](A)
        A = self.denseA(A)
        A = self.dropoutA(A, training=training)
        A = self.layernormA(A + Ain)
        return A

    def call(self, inputs, training=None, **kwargs):
        A=inputs[0]
        return self.ResUnit(A, training=training)

    def get_config(self):
        config=super().get_config()
        config = config.update({
                  "denseAs":self.denseAs,
                  "dropoutA":self.dropoutA,
                  "denseA":self.denseA,
                  "layernormA":self.layernormA,
                  "Adropout_rate": self.Adropout_rate,
                  "A_dff": self.A_dff,
                  "num_denseA":self.num_denseA
                  })
        return config



