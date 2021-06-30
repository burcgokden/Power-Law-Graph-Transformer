## Power Law Graph Transformer

This repository is the implementation of the Power Law Graph Transformer (PLGT) detailed in the research article: [Power Law Graph Transformer for Machine Translation and Representation Learning](https://github.com/burcgokden/Power-Law-Graph-Transformer/blob/main/plgt_paper.pdf)

Power Law Graph Transformer (PLGT) is a deductive-inductive transformer model that learns the power law relationships at the dataset level for the entire graph while providing predictions for graph instances the same way as a transductive transformer. It provides a new way to generalize and analyze data representations of graph structure of a dataset while keeping the same prediction capabilities of an attention based encoder-decoder model. 

PLGT uses a Power Law Graph Attention (PLGA) model to learn the graph representation of the dataset as its deductive output. PLGA learns the metric tensor and energy-curvature tensor for a graph instance usin linear self attention, non-linear power law distribution and deep residual neural networks. It replaces the Scaled Dot Product Attention (SDPA) in widely used transformer implementation. The PLGA is a generalized model developed over the screened Coulomb Attention model first implemented at [CoulGAT: A Graph Attention Framework with screened Coulomb Attention Mechanism](https://github.com/burcgokden/CoulGAT-Graph-Attention-Interpretability). The implementation here uses the training and evaluation framework for SDPA transformer implementation at [Transformer with Scaled Dot Product Attention](https://github.com/burcgokden/SDPA-Transformer-Wrapper) as a starting point.

#### Key Features:

- Capability to generalize graph structure at dataset level through learned power law parameters  of Power Law Graph Attention as deductive task outputs for representation learning.
- Capability to predict end-to-end in the same way as a transductive SDPA Transformer model as inductive task output.
- Flexible model customization through a hyperparameter dictionary for Power Law Graph Transformer model parameters for neural machine translation task.
- Simple interface for training the model with checkpoints at custom intervals, and highest accuracy observed.
- Early stopping after a number of epochs based on validation loss.
- Simple interface for evaluating trained model using BLEU score with greedy search and beam search.
- Data preparation framework for Neural Machine Translation for tensorflow datasets with capability to use a percentage of the train dataset or filter dataset based on a token  number in a sentence. 
- Capability to reverse source and target languages for input dataset.

#### Sample Run:

Sample run trains and evaluates a single layer 8-head PLGA Transformer model with 8 residual layers using a PT-EN translation task from tensorflow dataset found at: [ted_hrlr_translate/pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)

The tokenizer model is developed using BERT Subword Tokenizer for Machine Translation implemented at [BERT Subword Tokenizer for Machine Translation](https://github.com/burcgokden/BERT-Subword-Tokenizer-Wrapper)

- Prepare the Dataset for Machine Translation:

```python

import nmt_data_prep as ndp

inp_obj = ndp.src_tgt_data_prep(
                 src_lang='pt',
                 tgt_lang='en',
                 BUFFER_SIZE=20000,
                 BATCH_SIZE = 64,
                 dataset_file="ted_hrlr_translate/pt_to_en",
                 load_dataset=True,
                 train_percent=None,
                 model_name = "./ted_hrlr_translate_pt_en_tokenizer",
                 revert_order=False,
                 shuffle_set=True,
                 shuffle_files=True,
                 MAX_LENGTH=None, 
                 verbose=True)
```

- Define hyperparameter dictionary for PLGA Transformer:

```python
hpdict_plga_transformer = {
          "num_layers": 1,
          "d_model": 512,
          "num_heads": 8,
          "dropout_rate": 0.4,
          "dff": 2048,
          "att_dropout_rate_in":0.0,
          "att_dropout_rate_eij":0.1,                                                      
          "Adropout_rate":0.1,
          "A_dff":256,
          "num_reslayerA":8,
          "num_denseA":2,
          "input_vocab_size": inp_obj.tokenizers_src.get_vocab_size(), 
          "target_vocab_size": inp_obj.tokenizers_tgt.get_vocab_size(),
          "pe_input": 1000,
          "pe_target": 1000,
          "epochs": 120,
          "save_model_path": "my_plga_transformer",       
          "early_stop_threshold": 4.0,
          "early_stop_counter": 10,
          "early_stop_accuracy": 0.59,                                                                              
          "warmup_steps": 15000                                                                            
          }
```
- Initialize the end-to-end model training suite and run train:

```python
import plga_transformer_run_model as plga_run

e2e_obj=plga_run.plga_transformer_e2e(
                               tokenizer_obj_src = inp_obj.tokenizers_src,
                               tokenizer_obj_tgt = inp_obj.tokenizers_tgt,
                               checkpoint_path = "./model_saves/",
                               hpdict=hpdict_plga_transformer,
                               load_ckpt=None
                              )

train_loss, train_accuracy, val_loss, val_accuracy=e2e_obj.train_model(
                                                                       inp_obj.train_batches, 
                                                                       inp_obj.val_batches,
                                                                       chkpt_epochs=[40, 60, 70, 80, 90, 100, 110, 120]
                                                                      )

```

- Evaluate the trained SDPA Model using greedy or beam search:
```python
import plga_evaluate_bleu_score as ebg

#greedy search only
ebg.evaluate_bleu(
                 model_dict=hpdict_plga_transformer,
                 model_name="my_plga_transformer",
                 model_type='train',
                 src_lang='pt',
                 tgt_lang='en',
                 dataset_file="ted_hrlr_translate/pt_to_en",
                 revert_order=False,
                 inp_obj=None,
                 chkpt_path= './model_saves/',
                 data_path= './model_data/',              
                 load_ckpt='train', # 'val' | 'valacc' | custom checkpoint path
                 tok_model_name="./ted_hrlr_translate_pt_en_tokenizer",
                 max_length=50,  #offset to evaluate model beyond input sentence length
                 ds_max_length=None, #None for no filtering input sentence length
                 verbose=True
                )

#beam search
ebg.beam_evaluate_bleu(
                 model_dict=hpdict_plga_transformer,
                 beam_size=4,
                 model_name="my_plga_transformer",
                 model_type='train',
                 src_lang='pt',
                 tgt_lang='en',
                 dataset_file="ted_hrlr_translate/pt_to_en",
                 revert_order=False,
                 inp_obj=None,
                 chkpt_path= './model_saves/',
                 data_path= './model_data/',              
                 load_ckpt='train',
                 tok_model_name="./ted_hrlr_translate_pt_en_tokenizer",
                 max_length=50,
                 ds_max_length=None,
                 verbose=True
                )

```
#### Deductive Task Outputs:

The deductive task outputs can be obtained by evaluating an input sentence from a model checkpoint:

```python

e2e_model=plga_run.plga_transformer_e2e(
                                        tokenizer_obj_src = inp_obj.tokenizers_src,
                                        tokenizer_obj_tgt = inp_obj.tokenizers_tgt,
                                        checkpoint_path = "./model_saves/",
                                        hpdict=hpdict_plga_transformer,
                                        load_ckpt='train' # 'val' | 'valacc' | custom checkpoint path
                                       )
                                       
sentence = "este é um problema que temos que resolver ."
ground_truth = "this is a problem we have to solve ."

attention_save_path = "./attention_weights/my_attention.pkl"
translated_text, translated_tokens, att_weights, eval_max_len = e2e_model.evaluate(sentence, max_length=50,
                                                                                  save_att=attention_save_path)
e2e_model.print_translation(sentence, translated_text, ground_truth, eval_max_len)


```

The attention weight output contains the following data:

att_weights: Is a list of [Source LM attention weights, Target LM attention weights, X-LM attention weights]
att_weights[i][0]: Is a list of [**E<sub>LM</sub>**,** A<sub>LM</sub>**, **P<sub>LM</sub>**, **a<sub>LM</sub>**, **b<sub>a</sub>**, **G<sub>LM</sub>**, **E<sub>LM</sub>**(unmasked)] for i=0,1,2 (SLM, TLM, XLM)
att_weights[i][0][1][0][0].numpy() : **A<sub>LM</sub>** is an array of [num_head, d<sub>k</sub>, d<sub>k</sub>]
att_weights[i][0][2][0].numpy() : **P<sub>LM</sub>** is an array of [num_head, d<sub>k</sub>, d<sub>k</sub>]



