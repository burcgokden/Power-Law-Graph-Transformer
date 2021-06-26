
import os
import time

import numpy as np
import tensorflow as tf

import plga_transformer_model as plga_model
import common as cm

class plga_transformer_e2e:
    '''
    Trains and evaluates power law graph transformer model.
    '''

    def __init__(self, tokenizer_obj_src, tokenizer_obj_tgt, hpdict=None,
                 checkpoint_path = "./plga_checkpoints/", load_ckpt=None):

        if hpdict:
            self.hpdict = hpdict
        else:
            print("USING DEFAULT HYPERPARAMETERS")
            self.hpdict={"num_layers": 1,
                         "d_model": 512,
                         "num_heads": 8,
                         "dropout_rate": 0.4,
                         "dff": 2048,
                         "att_dropout_rate_in": 0.0,
                         "att_dropout_rate_eij": 0.1,
                         "Adropout_rate":0.1,
                         "num_reslayerA":8,
                         "num_denseA":2,
                         "A_dff":256,
                         "input_vocab_size": tokenizer_obj_src.get_vocab_size(),
                         "target_vocab_size" : tokenizer_obj_tgt.get_vocab_size(),
                         "pe_input":1000,
                         "pe_target":1000,
                         "epochs":80,
                         "save_model_path": "./default_plga_transformer", #name to save parameters and checkpoints with. This is file name
                         "early_stop_threshold":4.0, #ceiling value for early stop loss. None disables early stop
                         "early_stop_counter": 10, #number of epochs before a checkpoint is saved. None is no limit.
                         "earl_stop_accuracy": 0.50,  # Accuracy threshold to start saving model
                         "warmup_steps": 15000
            }
        print(f"hyperparameters are {self.hpdict}")

        #tokenizer for source language
        self.tokenizers_src=tokenizer_obj_src
        #tokenizer for target language
        self.tokenizers_tgt=tokenizer_obj_tgt

        self.learning_rate = CustomSchedule(self.hpdict["d_model"], warmup_steps=self.hpdict["warmup_steps"])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_function=masked_loss_function()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

        self.transformer = plga_model.Transformer(
            num_layers=self.hpdict["num_layers"],
            d_model=self.hpdict["d_model"],
            num_heads=self.hpdict["num_heads"],
            dff=self.hpdict["dff"],
            input_vocab_size=self.hpdict["input_vocab_size"],
            target_vocab_size=self.hpdict["target_vocab_size"],
            pe_input=self.hpdict["pe_input"],
            pe_target=self.hpdict["pe_target"],
            rate=self.hpdict["dropout_rate"],
            att_dropout_rate_in=self.hpdict["att_dropout_rate_in"],
            att_dropout_rate_eij=self.hpdict["att_dropout_rate_eij"],
            Adropout_rate=self.hpdict["Adropout_rate"],
            A_dff=self.hpdict["A_dff"],
            num_reslayerA=self.hpdict["num_reslayerA"],
            num_denseA=self.hpdict["num_denseA"],
        )

        self.checkpoint_path = checkpoint_path
        self.train_ckpt_path=os.path.join(self.checkpoint_path, "train", hpdict["save_model_path"])
        self.val_ckpt_path=os.path.join(self.checkpoint_path, "validate", hpdict["save_model_path"])
        self.valacc_ckpt_path = os.path.join(self.checkpoint_path, "validate_acc", hpdict["save_model_path"])

        if not os.path.isdir(self.train_ckpt_path):
            print(f"Creating train ckpt dir: {self.train_ckpt_path}")
            os.makedirs(self.train_ckpt_path)
            cm.pklsave(os.path.join(self.train_ckpt_path, hpdict["save_model_path"] + "_hparams.pkl"), self.hpdict)
        if not os.path.isdir(self.val_ckpt_path):
            print(f"Creating val ckpt dir: {self.val_ckpt_path}")
            os.makedirs(self.val_ckpt_path)
            cm.pklsave(os.path.join(self.val_ckpt_path, hpdict["save_model_path"] + "_hparams.pkl"), self.hpdict)
        if not os.path.isdir(self.valacc_ckpt_path):
            print(f"Creating val accuracy ckpt dir: {self.valacc_ckpt_path}")
            os.makedirs(self.valacc_ckpt_path)
            cm.pklsave(os.path.join(self.valacc_ckpt_path, hpdict["save_model_path"] + "_hparams.pkl"), self.hpdict)

        train_ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        val_ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        valacc_ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)

        self.train_ckpt_manager = tf.train.CheckpointManager(train_ckpt,
                                                             directory=self.train_ckpt_path,
                                                             checkpoint_name="train_"+hpdict["save_model_path"],
                                                             max_to_keep=10)
        self.val_ckpt_manager = tf.train.CheckpointManager(val_ckpt,
                                                           directory=self.val_ckpt_path,
                                                           checkpoint_name="val_"+hpdict["save_model_path"],
                                                           max_to_keep=3)
        self.valacc_ckpt_manager = tf.train.CheckpointManager(valacc_ckpt,
                                                           directory=self.valacc_ckpt_path,
                                                           checkpoint_name="valacc_"+hpdict["save_model_path"],
                                                           max_to_keep=3)

        # if a checkpoint exists, restore the latest checkpoint.
        if load_ckpt=="train":
            if self.train_ckpt_manager.latest_checkpoint:
                train_ckpt.restore(self.train_ckpt_manager.latest_checkpoint)
                print('Latest train checkpoint restored!!')
        elif load_ckpt=="val":
            if self.val_ckpt_manager.latest_checkpoint:
                val_ckpt.restore(self.val_ckpt_manager.latest_checkpoint)
                print('Latest val checkpoint restored!!')
        elif load_ckpt=="valacc":
            if self.valacc_ckpt_manager.latest_checkpoint:
                valacc_ckpt.restore(self.valacc_ckpt_manager.latest_checkpoint)
                print('Latest val accuracy checkpoint restored!!')
        elif load_ckpt is None:
            print('Checkpoint restoration is skipped')
        else:
            print("Attempting to restore the checkpoint path specified.")
            train_ckpt.restore(load_ckpt)


    def create_masks(self, inp, tar):
        '''
        Create masks for encoder and decoder.
        '''

        #Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        # dec_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        '''
        Train step for single token for transformer model.
        '''

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer([inp, tar_inp,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask], training=True)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(self.accuracy_function(tar_real, predictions))

    def train_model(self, train_batches, val_batches, chkpt_epochs=None):
        '''
        Method for training model for multiple epochs.
        '''

        EPOCHS=self.hpdict["epochs"]
        #set epochs to save as checkpoint for 60% and 75% of training epochs. (Max ckpt number - 1) elements
        chkpt_epochs= chkpt_epochs if chkpt_epochs is not None else [int(np.floor(EPOCHS*0.34)), int(np.floor(EPOCHS*0.5)), int(np.floor(EPOCHS*0.6)), int(np.floor(EPOCHS*0.75))]
        print(f"Train checkpoints are at epochs: {chkpt_epochs}")

        #initialize lists to collect loss and accuracy data per epoch
        train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst=[],[],[],[]
        early_stop_loss = self.hpdict["early_stop_threshold"]
        early_stop_accuracy = self.hpdict["early_stop_accuracy"]
        post_early_stop_epoch = None
        for epoch in range(EPOCHS):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_batches):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')

            if (epoch + 1) in chkpt_epochs:
                ckpt_save_path = self.train_ckpt_manager.save()
                print(f'Saving train checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            train_loss_lst.append(self.train_loss.result().numpy())
            train_acc_lst.append(self.train_accuracy.result().numpy())

            self.validate_model(val_batches, epoch)
            val_loss_lst.append(self.val_loss.result().numpy())
            val_acc_lst.append(self.val_accuracy.result().numpy())

            if early_stop_loss is not None:
                if early_stop_loss > self.val_loss.result().numpy():
                    post_early_stop_epoch = 0
                    early_stop_loss = self.val_loss.result().numpy()
                elif self.hpdict["early_stop_counter"] is not None:
                    if (post_early_stop_epoch is not None):
                        post_early_stop_epoch += 1
                        if post_early_stop_epoch >= self.hpdict["early_stop_counter"]:
                            print(f"Saving checkpoint after {post_early_stop_epoch} epochs as validation checkpoint.")
                            val_ckpt_save_path = self.val_ckpt_manager.save()
                            print(f'Saving val checkpoint for epoch {epoch + 1} at {val_ckpt_save_path}')
                            post_early_stop_epoch = None
            if early_stop_accuracy is not None:
                if early_stop_accuracy < self.val_accuracy.result().numpy():
                    early_stop_accuracy = self.val_accuracy.result().numpy()
                    print(f"Saving checkpoint at epoch {epoch+1} at accuracy {early_stop_accuracy*100:.2f}% as accuracy validation checkpoint.")
                    valacc_ckpt_save_path = self.valacc_ckpt_manager.save()
                    print(f'Saving val accuracy checkpoint for epoch {epoch + 1} at {valacc_ckpt_save_path}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        #save loss and accuracy data for train and validation runs
        cm.pklsave(self.train_ckpt_path+'/train_loss.pkl', train_loss_lst)
        cm.pklsave(self.train_ckpt_path+'/val_loss.pkl', val_loss_lst)
        cm.pklsave(self.train_ckpt_path+'/train_accuracies.pkl', train_acc_lst)
        cm.pklsave(self.train_ckpt_path+'/val_accuracies.pkl', val_acc_lst)

        final_ckpt_save_path = self.train_ckpt_manager.save()
        print(f'Saving final train checkpoint for epoch {EPOCHS} at {final_ckpt_save_path}')

        return train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst

    def validate_model(self, val_batches, epoch):
        '''
        This runs the model on val dataset and returns loss and accuracy during training.
        Args:
            val_batches: the validation batches same size as train batches
        Returns:
            loss: the loss averaged over all batches
            accuracy: the accuracy averaged over all batches
        '''

        loss_lst=[]
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(val_batches):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

            predictions, _ = self.transformer([inp, tar_inp,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask], training=False)
            loss = self.loss_function(tar_real, predictions)
            loss_lst.append(loss)

            self.val_loss(loss)
            self.val_accuracy(self.accuracy_function(tar_real, predictions))

        mean_loss = np.mean(loss_lst)
        print(f'Epoch {epoch + 1} Val Loss {self.val_loss.result():.4f} Val Accuracy {self.val_accuracy.result():.4f}')

        return self.val_loss.result(), self.val_accuracy.result(), mean_loss

    def evaluate(self, sentence, max_length=50, save_att=None):
        '''
        Evaluate input sentence using greedy search.
        Args:
            sentence: source sentence as input.
            max_length: maximum number of iterations to run.
            save_att: path location to save attention weights.
        Returns:
            Predicted text, tokens and attention weights.
        '''

        sentence = tf.convert_to_tensor([sentence])
        sentence = self.tokenizers_src.tokenize(sentence).to_tensor()
        eval_max_length=tf.shape(sentence)[1].numpy()+max_length

        encoder_input = sentence
        start, end = self.tokenizers_tgt.tokenize([''])[0]
        output = tf.convert_to_tensor([start])
        output = tf.expand_dims(output, 0)
        att_weights=None

        for i in range(eval_max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, att_weights = self.transformer([encoder_input,
                                                         output,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            output = tf.concat([output, predicted_id], axis=-1)

            if predicted_id == end:
                break

        text = self.tokenizers_tgt.detokenize(output)[0]

        tokens = self.tokenizers_tgt.lookup(output)[0]

        #save attention weights
        if save_att is not None:
            print("saving attention weights")
            cm.pklsave(save_att, att_weights)

        return text, tokens, att_weights, eval_max_length

    def evaluate_test(self, test_inp, test_ref=None, max_length=50, filename=None, verbose=None):
        '''
        This method evaluates the test input consisting of a list of sentences.
        These sentences are then saved for further processing for token generation.
        Args:
            test_inp: list of input sentences from source language.
            test_ref: list of ground truth sentences
            max_length: maximum number of iterations to run autoregressively.
            filename: file path to save the predicted sentences and tokens.
            verbose: If set True, predicted and ground sentence are printed for each input sentence.
        Returns:
            Saves predicted sentences and tokens.
        '''
        start = time.time()
        predicted_sentences=[]
        predicted_tokens=[]
        tot_len=len(test_inp)
        for cnt, inp in enumerate(test_inp):
            ps, ptk, _, max_eval_length = self.evaluate(inp, max_length=max_length)
            ps = ps.numpy().decode('utf-8')
            ptk = ptk.numpy()
            if verbose:
                print(f"{cnt+1}/{tot_len}: Max Eval Length is {max_eval_length}")
                print(f"Translating from: {inp}")
                print(f"Translated to: {ps}")
                if test_ref is not None:
                    print(f"Ground Truth: {test_ref[cnt]}")
                print('')
            predicted_sentences.append(ps)
            predicted_tokens.append(ptk)
            if (cnt+1) % 50 == 0:
                print(f"{cnt+1}/{tot_len} predictions are done.")
                print(f"Last predicted sentence is: {ps}")
                print(f"Last predicted tokens are: {ptk}")
                print(f'Time taken for {cnt+1} predictions: {time.time() - start:.2f} secs\n\n')
        print(f"Predictions are complete. {tot_len} sentences are predicted")
        if filename is not None:
            cm.pklsave(filename, [predicted_sentences, predicted_tokens])
            print(f"[predicted_sentences, predicted_tokens] are saved in {filename}")

        return predicted_sentences, predicted_tokens


    def beam_evaluate(self, sentence, beam_size=4, max_length=50):
        '''
        A beam search implementation for transformer model.
        '''

        sentence = tf.convert_to_tensor([sentence])
        sentence = self.tokenizers_src.tokenize(sentence).to_tensor()
        eval_max_length = tf.shape(sentence)[1].numpy() + max_length

        encoder_input = sentence
        start, end = self.tokenizers_tgt.tokenize([''])[0]

        beam_out = [([start.numpy()], 0.0)]
        finalized_seq = []  # for sequences that saw eos

        # generate predictions
        for i in range(eval_max_length):
            if not beam_out:
                break
            new_sequences=[]
            for outseq, outval in beam_out:
                output = tf.convert_to_tensor(outseq, dtype=tf.int64)
                output = tf.expand_dims(output, 0)
                enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)

                # predictions.shape == (batch_size, seq_len, vocab_size)
                predictions,_ = self.transformer([encoder_input, output,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask], training=False)
                # select the last word from the seq_len dimension
                predictions = tf.nn.softmax(predictions[:, -1:, :], axis=-1)  # (batch_size, 1, vocab_size)
                top_k_predvals, top_k_predids = tf.math.top_k(predictions, k=beam_size)  # [batch_size,1, beam_size]

                top_k_predval_all = top_k_predvals.numpy()
                top_k_predid_all = top_k_predids.numpy()
                top_k_predval_all = top_k_predval_all[0][0]
                top_k_predid_all = top_k_predid_all[0][0]

                old_seq, old_score = outseq, outval
                len_norm_oldseq=((5.0+len(old_seq))**0.6)/(6.0**0.6)
                tot_old_score=old_score*len_norm_oldseq
                # for every seq in output sequences check if it maximizes the prediction
                for predid, predval in zip(top_k_predid_all, top_k_predval_all):
                    new_seq = old_seq + [predid]
                    len_norm_newseq = ((5.0 + len(new_seq))**0.6) /(6.0**0.6)
                    # calculate log likelihood to minimize for best case
                    new_score = (tot_old_score - np.log(predval))/len_norm_newseq
                    if predid != end.numpy():
                        new_sequences.append([new_seq, new_score])
                    else:
                        finalized_seq.append([new_seq, new_score])

            beam_out = sorted(new_sequences, key=lambda x: x[1])
            beam_out = beam_out[:beam_size]

        finalized_seq.extend(new_sequences)
        finalized_seq.sort(key=lambda x: x[1])

        # detokenize each beam and create tokens.
        beam_tokens, beam_detokenized = [], []
        for t in finalized_seq[:beam_size]:
            t = tf.convert_to_tensor([t[0]], dtype=tf.int64)
            beam_detokenized.append(self.tokenizers_tgt.detokenize(t)[0])
            beam_tokens.append(self.tokenizers_tgt.lookup(t)[0])

        return beam_detokenized, beam_tokens, finalized_seq[:beam_size], eval_max_length


    def beam_evaluate_test(self, test_inp, test_ref=None, beam_size=4,  max_length=50, filename=None, verbose=None):
        '''
        Evaluate list of test sentences using beam_evaluate method.
        '''
        start = time.time()
        predicted_sentences=[]
        predicted_tokens=[]
        predicted_beam_seq=[]
        tot_len=len(test_inp)
        unpred_lst=[]
        for cnt, inp in enumerate(test_inp):
            bps, bptk, bfs, max_eval_length = self.beam_evaluate(inp, beam_size=beam_size, max_length=max_length)
            npbps, npbptk = [], [] #temporary lists to keep converted predicted sentences and tokens.
            for psentence in bps:
                psentence = psentence.numpy().decode('utf-8') #convert to numpy string
                npbps.append(psentence)
            for ptoken in bptk:
                ptoken = ptoken.numpy()
                npbptk.append(ptoken)
            if verbose:
                print(f"{cnt+1}/{tot_len}: Max Eval Length is {max_eval_length}")
                print(f"Translating from: {inp}")
                if len(npbps) > 0:
                    print(f"Best probable translation: {npbps[0]}")
                else:
                    print(f"WARNING: A PREDICTION WAS NOT FOUND WITHIN EVAL LENGTH OF {max_length} WITH BEAM_SIZE: {beam_size} FOR INP: {inp}")
                    unpred_lst.append(inp)
                if test_ref is not None:
                    print(f"Ground Truth: {test_ref[cnt]}")
                print('') #extra space
            predicted_sentences.append(npbps) #each element is list of beam_size numpy sentences
            predicted_tokens.append(npbptk) #each element is list of beam_size numpy token lists
            predicted_beam_seq.append(bfs) #each element is list of beam_size numpy (token id seq, beam_score)
            if (cnt+1) % 50 == 0:
                print(f"{cnt+1}/{tot_len} predictions are done.")
                if len(npbps) > 0:
                    print(f"Last predicted sentence is: {npbps[0]}")
                    print(f"Last predicted tokens are: {npbptk[0]}")
                else:
                    print(f"WARNING: A PREDICTION WAS NOT FOUND WITHIN EVAL LENGTH OF {max_length} WITH BEAM_SIZE: {beam_size} FOR INP: {inp}")
                print(f'Time taken for {cnt+1} predictions: {time.time() - start:.2f} secs\n\n')
        print(f"Predictions are complete. {tot_len} sentences are predicted")
        print(f'Time taken for {tot_len} predictions: {time.time() - start:.2f} secs\n\n')
        if filename is not None:
            cm.pklsave(filename, [predicted_sentences, predicted_tokens, predicted_beam_seq, unpred_lst])
            print(f"[predicted_sentences, predicted_tokens, predicted_beam_seq] are saved in {filename}")

        return predicted_sentences, predicted_tokens, predicted_beam_seq, unpred_lst

    @staticmethod
    def print_translation(sentence, tokens, ground_truth, max_eval_length):
        print(f"Max Eval Length: {max_eval_length}")
        print(f'{"Input:":15s}: {sentence}')
        print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
        print(f'{"Ground truth":15s}: {ground_truth}')

    @staticmethod
    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step, **kwargs):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config=super().get_config()
        config=config.update({
            "d_model": self.d_model,
            "warmup_steps":self.warmup_steps
        })
        return config

class masked_loss_function(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true,0))
        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_*=mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def get_config(self):
        config = super().get_config()
        return config
