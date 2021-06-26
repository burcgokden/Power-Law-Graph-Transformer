
import logging

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text

logging.getLogger('tensorflow').setLevel(logging.ERROR)


class src_tgt_data_prep:
    '''
    Prepares data for encoder-decoder architecture for machine translation task
    default inputs are for portuguese to english dataset from TED Talks Open Translation Project.
    '''
    def __init__(self,
                 src_lang='pt',
                 tgt_lang='en',
                 BUFFER_SIZE=20000,
                 BATCH_SIZE = 64,
                 dataset_file='ted_hrlr_translate/pt_to_en',
                 load_dataset=True,
                 train_percent=None,
                 model_name = "./ted_hrlr_translate_pt_en_tokenizer",
                 revert_order=False,
                 shuffle_set=True,
                 shuffle_files=True,
                 MAX_LENGTH=None,
                 verbose=False):
        '''
        This init method asks for tokenizer source and target object loaded and ready to provide.
        The dataset may have order reverted, this method does the conversion to intended source target order.

        Args:
            src_lang: source language abbreviation as string
            tgt_lang: target language abbreviation as string
            BUFFER_SIZE: Buffer size for shuffling
            BATCH_SIZE: batch size for dataset
            dataset_file: path to tensorflow dataset
            load_dataset: if True load the dataset
            train_percent: Percentage of train data to be loaded. 1-100. None loads all training data.
            model_name: file path for tokenizer model.
            revert_order: If True, it reverts the order of language pairs in dataset_file. Reverted order should match
                          src_lang/tgt_lang assignment.
            shuffle_set:If True shuffle the dataset while loading
            shuffle_files: shuffle dataset files while loading
            MAX_LENGTH: Maximum number of tokens in each sentence.
            verbose: If True print out more details.

        Returns batched, tokenized, filtered train, validation datasets and test dataset. Tokenizer methods are accessible
        through instance of this class object
        '''

        self.BUFFER_SIZE=BUFFER_SIZE
        self.BATCH_SIZE=BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH
        self.model_name = model_name
        self.revert_order=revert_order
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizers_src, self.tokenizers_tgt, self.tokenizers = self.load_tokenizer()

        #load dataset
        if load_dataset:
            print("LOADING DATASET")
            if train_percent:
                #load only percentage of train data
                examples, metadata = tfds.load(dataset_file,
                                                split=[f'train[:{train_percent}%]', 'validation', 'test'],
                                                with_info=True, as_supervised=True, shuffle_files=shuffle_files)
            else:
                #load all data
                examples, metadata = tfds.load(dataset_file,
                                               split=['train', 'validation', 'test'],
                                               with_info=True, as_supervised=True, shuffle_files=shuffle_files)

            if self.revert_order:
                #revert the order if intended source and target language orders are reversed
                #tokenizer source and tokenizer target are intended values.
                print(f"REVERTING ORDER OF DATASET TUPLES TO (SRC, TGT) : {self.src_lang},{self.tgt_lang}")
                self.train_examples = examples[0].map(lambda dsl1, dsl2: [dsl2, dsl1])
                self.val_examples = examples[1].map(lambda dsl1, dsl2: [dsl2, dsl1])
                self.test_examples=examples[2].map(lambda dsl1, dsl2: [dsl2, dsl1])
                self.examples = examples
                self.metadata = metadata
            else:
                print(f"ORDER OF DATASET TUPLES (SRC, TGT) : {self.src_lang},{self.tgt_lang}")
                self.train_examples = examples[0]
                self.val_examples = examples[1]
                self.test_examples=examples[2]
                self.examples=None
                self.metadata=metadata
        else:
            print("SKIPPED LOADING DATASET")

        #print some info about tokenizer model
        load_tokenizer_model= self.tokenizers_src and self.tokenizers_tgt
        if load_tokenizer_model:
            print("SOURCE AND TARGET TOKENIZERS INFO")
            print(f"Methods for source lang: {self.src_lang}")
            print([item for item in dir(self.tokenizers_src) if not item.startswith('_')])
            print((f"Methods for tgt lang: {self.tgt_lang}"))
            print([item for item in dir(self.tokenizers_tgt) if not item.startswith('_')])
        else:
            print("PLEASE PROVIDE TOKENIZERS CORRECTLY")

        if self.MAX_LENGTH is None:
            #create batched and tokenized datasets.
            print("CREATING SHUFFLED BATCHED DATASETS FOR TRAINING AND VALIDATION")
            self.train_batches=self.make_batches(self.train_examples, map_tokenize=load_tokenizer_model, shuffle_set=shuffle_set)
            self.val_batches=self.make_batches(self.val_examples, map_tokenize=load_tokenizer_model, shuffle_set=False)
            self.test_examples = self.test_examples.prefetch(tf.data.AUTOTUNE)
        else:
            self.train_batches=self.make_padded_batches(self.train_examples, shuffle_set=shuffle_set)
            self.val_batches=self.make_padded_batches(self.val_examples, shuffle_set=False)
            self.test_examples=self.filter_test(self.test_examples)
            if verbose:
                #these operations are very slow so for large datasets should be avoided.
                print(f"FILTERED BATCHED TRAIN DATASET ELEMENT COUNT: {self.dataset_batch_cardinality(self.train_batches)*self.BATCH_SIZE}")
                print(f"FILTERED BATCHED VAL DATASET ELEMENT COUNT: {self.dataset_batch_cardinality(self.val_batches)*self.BATCH_SIZE}")

    @staticmethod
    def dataset_batch_cardinality(ds):
        cnt = 0
        for _ in ds:
            cnt += 1
        return cnt

    def filter_test(self, test_ds):
        '''
        The test needs to be first tokenized,
        filter for token length and then detokenized.
        '''

        print(f"ORIGINAL TEST DATASET LENGTH: {len(test_ds)}")

        test_ds=test_ds.batch(1).map(self.tokenize_pairs_src_tgt)
        test_ds=test_ds.unbatch().filter(self.filter_max_length)
        test_ds=test_ds.batch(1).map(self.detokenize_pairs_src_tgt)
        test_ds=test_ds.unbatch().prefetch(tf.data.AUTOTUNE)

        for ts in test_ds.take(3):
            print(f"DETOKENIZED TEST SAMPLE LESS THAN LENGTH {self.MAX_LENGTH}: {ts}")
        print(f"FILTERED TEST LENGTH: {self.dataset_batch_cardinality(test_ds)}")

        return test_ds

    def detokenize_pairs_src_tgt(self, src, tgt):

        src = self.tokenizers_src.detokenize(src)
        tgt = self.tokenizers_tgt.detokenize(tgt)

        return src, tgt



    def load_tokenizer(self):
        '''
        Run this first to get tokenizers pairs for intended source and target language.
        Returns source tokenizer, target tokenizer and tokenizer object
        '''
        print(f"LOADING TOKENIZER AT {self.model_name}")
        tokenizers = tf.saved_model.load(self.model_name)
        print("THE TOKENIZER LANGUAGES AVAILABLE ARE:")
        print([item for item in dir(tokenizers) if not item.startswith('_')])
        tokenizers_src=getattr(tokenizers, self.src_lang, None)
        tokenizers_tgt=getattr(tokenizers, self.tgt_lang, None)

        return tokenizers_src, tokenizers_tgt, tokenizers



    def tokenize_pairs_src_tgt(self, src, tgt):
        '''
        Use tokenizer model to create tokenized pairs.
        '''
        src = self.tokenizers_src.tokenize(src)
        # Convert from ragged to dense, padding with zeros.
        src = src.to_tensor()

        tgt = self.tokenizers_tgt.tokenize(tgt)
        # Convert from ragged to dense, padding with zeros.
        tgt = tgt.to_tensor()

        return src, tgt

    def make_batches(self, ds, map_tokenize=True, shuffle_set=True):
        '''
        method to create dataset batches and map each element with tokenizer model
        it takes a dataset that contains lang1, lang2 pairs.
        '''
        #shuffle dataset and make batches
        ds_batched=ds
        if shuffle_set:
            ds_batched = ds_batched.shuffle(self.BUFFER_SIZE)

        ds_batched=ds_batched.batch(self.BATCH_SIZE)
        if map_tokenize:
            ds_batched = ds_batched.map(self.tokenize_pairs_src_tgt, num_parallel_calls=tf.data.AUTOTUNE)

        ds_batched=ds_batched.prefetch(tf.data.AUTOTUNE)
        print("Dataset element spec:", ds_batched.element_spec)

        return ds_batched

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x) <= self.MAX_LENGTH,
                              tf.size(y) <= self.MAX_LENGTH)

    def make_padded_batches(self, ds, shuffle_set=True):
        '''
        If a max length is specified, the dataset is filtered, padded then batched.
        '''

        ds_batched = ds.batch(1)
        ds_batched = ds_batched.map(self.tokenize_pairs_src_tgt, num_parallel_calls=tf.data.AUTOTUNE)
        ds_batched=ds_batched.unbatch()
        if shuffle_set:
            ds_batched=ds_batched.shuffle(self.BUFFER_SIZE)
        ds_batched=ds_batched.filter(self.filter_max_length).padded_batch(self.BATCH_SIZE, padded_shapes=(self.MAX_LENGTH, self.MAX_LENGTH))
        ds_batched = ds_batched.prefetch(tf.data.AUTOTUNE)

        return ds_batched

def download_tokenizer_model(model_name = "ted_hrlr_translate_pt_en_converter", cache_dir="."):
    '''
    Downloads a pretrained tokenizer model to a cache dir where model can be loaded from.
    Can be used once to download the model. model_name needs to match exactly the name of the model.
    '''

    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir=cache_dir, cache_subdir='', extract=True
    )
