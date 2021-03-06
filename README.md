## Power Law Graph Transformer

This repository is the implementation of the Power Law Graph Transformer (PLGT) detailed in the research article: [Power Law Graph Transformer for Machine Translation and Representation Learning](https://arxiv.org/abs/2107.02039)

Power Law Graph Transformer is a deductive-inductive transformer model that learns the power law relationships at the dataset level for the entire graph while providing predictions for graph instances the same way as a transductive transformer. It provides a new way to generalize and analyze data representations of graph structure of a dataset while keeping the same prediction capabilities of an attention based encoder-decoder model. 

PLGT uses a Power Law Graph Attention (PLGA) model to learn the graph representation of the dataset as its deductive output. PLGA learns the metric tensor and energy-curvature tensor for a graph instance using linear self attention, non-linear power law distribution and deep residual neural networks. It replaces the Scaled Dot Product Attention (SDPA) in widely used transformer implementation. The PLGA is a generalized model developed over the screened Coulomb Attention model first implemented at [CoulGAT: A Graph Attention Framework with screened Coulomb Attention Mechanism](https://github.com/burcgokden/CoulGAT-Graph-Attention-Interpretability). The implementation here uses the training and evaluation framework for SDPA transformer implementation at [Transformer with Scaled Dot Product Attention](https://github.com/burcgokden/SDPA-Transformer-Wrapper) as a starting point.

#### Key Features:

- Capability to generalize graph structure at dataset level through learned power law parameters of Power Law Graph Attention as deductive task outputs for representation learning.
- Capability to predict end-to-end in the same way as a transductive SDPA Transformer model as inductive task output.
- Flexible model customization through a hyperparameter dictionary for Power Law Graph Transformer model parameters for neural machine translation task.
- Simple interface for training the model with checkpoints at custom intervals, and highest accuracy observed.
- Early stopping after a number of epochs based on validation loss.
- Simple interface for evaluating trained model using BLEU score with greedy search and beam search.
- Data preparation framework for Neural Machine Translation for tensorflow datasets with capability to use a percentage of the train dataset or filter dataset based on a token  number in a sentence. 
- Capability to reverse source and target languages for input dataset.
- Keeps track of train and validation loss/accuracy for each epoch.

#### Sample Run:

Sample run trains and evaluates a single layer 8-head PLGA Transformer model with 8 residual layers for PT-EN translation task from tensorflow dataset found at: [ted_hrlr_translate/pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)

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

- Evaluate the trained PLGA Model using greedy or beam search:
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
                                       
sentence = "este ?? um problema que temos que resolver ."
ground_truth = "this is a problem we have to solve ."

attention_save_path = "./attention_weights/my_attention.pkl"
translated_text, translated_tokens, att_weights, eval_max_len = e2e_model.evaluate(sentence, max_length=50,
                                                                                  save_att=attention_save_path)
e2e_model.print_translation(sentence, translated_text, ground_truth, eval_max_len)


```

The attention weight output contains the following data:

- att_weights: Is a list of [Source LM attention weights, Target LM attention weights, X-LM attention weights]
- att_weights[i][0]: Is a list of [**E<sub>LM</sub>**, **A<sub>LM</sub>**, **P<sub>LM</sub>**, **a<sub>LM</sub>**, **b<sub>a</sub>**, **G<sub>LM</sub>**, **E<sub>LM</sub>**(unmasked)] for i=0,1,2 (SLM, TLM, XLM)
- att_weights[i][0][1][0][0].numpy() : **A<sub>LM</sub>** is an array of [num_head, d<sub>k</sub>, d<sub>k</sub>]
- att_weights[i][0][2][0].numpy() : **P<sub>LM</sub>** is an array of [num_head, d<sub>k</sub>, d<sub>k</sub>]

#### Loss and Accuracy Curves:

Train and validation loss/accuracy values for each epoch are saved as pickle file and can be found in the train folder under save_model_path name:

```python
import common as cm

train_loss=cm.pklload("./model_saves/train/my_plga_transformer/train_loss.pkl")
val_loss=cm.pklload("./model_saves/train/my_plga_transformer/val_loss.pkl")
train_acc=cm.pklload("./model_saves/train/my_plga_transformer/train_accuracies.pkl")
val_acc=cm.pklload("./model_saves/train/my_plga_transformer/val_accuracies.pkl")
```

#### Single Instance Evaluation:

A sentence can be translated and compared to ground truth using greedy search only or beam search methods for single instance evaluation:

```python
#greedy search only
translated_text, translated_tokens, _, eval_length = e2e_model.evaluate(sentence, max_length=50)
e2e_model.print_translation(sentence, translated_text, ground_truth, eval_length)

#beam search
translated_text_list, translated_tokens_list, tranlated_tokenid_list, eval_length = e2e_model.beam_evaluate(sentence, beam_size=4, max_length=50)
e2e_model.print_translation(sentence, translated_text_list[0], ground_truth, eval_length)
```

- Below sentences from test dataset are evaluated with beam length=4 by a model trained with same hyperparameters as model #2 detailed in the research article. Evaluation output may vary with each newly trained model.

>**Translating from:** perdemos o medo de criar uma coisa nova .  
>**Best probable translation:** we lost the fear of creating something new .  
>**Ground Truth:** we lost the fear of creating something new .
>
> **Translating from:** vou mostrar aqui alguns exemplos , e vamos examinar alguns deles .  
> **Best probable translation:** so i ' m going to show you a few examples , and we ' re going to examine some of them .  
> **Ground Truth:** i 'm going to show you some examples here , and we will run through some of them .  
>
> **Translating from:** ok , hoje quero falar sobre a forma como falamos do amor .  
> **Best probable translation:** okay , today , i want to talk about how we talk about love .  
> **Ground Truth:** ok , so today i want to talk about how we talk about love .  
>
> **Translating from:** mas h?? uma grande diferen??a , isso s?? acontece dentro da col??nia .  
> **Best probable translation:** but there ' s a big difference , that only happens within the colony .  
> **Ground Truth:** but there 's a big difference , which is that it only happens within the colony .  
>
> **Translating from:** mas muito bons a absorver informa????o de muitas fontes diversas ao mesmo tempo .  
> **Best probable translation:** but very good at absorbing information from many sources at the same time .  
> **Ground Truth:** but they 're very good at taking in lots of information from lots of different sources at once .  
>
> **Translating from:** n??o podia construir isto com um anel de a??o , da forma que sabia .  
> **Best probable translation:** i could n ' t build this with a steel ring , the way i knew .  
> **Ground Truth:** i could n't build this with a steel ring , the way i knew .  
>
> **Translating from:** e gostaria de continuar a construir monumentos , que s??o amados por pessoas .  
> **Best probable translation:** and i ' d like to continue building monuments , which are loved by people .  
> **Ground Truth:** and i 'd like to keep building monuments that are beloved by people .  
>
> **Translating from:** a quest??o ?? que temos que ter um contexto , um limite para as nossas a????es em tudo isto .  
> **Best probable translation:** the question is that we have to have a context , a range for our actions in everything .  
> **Ground Truth:** the point is that we have to have a context , a gauge for our actions in all this .  
>
> **Translating from:** somos mais inteligentes , mais flexiv??is , capazes de aprender mais , sobrevivemos em diferentes ambientes , emigr??mos para povoar o mundo e viaj??mos at?? ao espa??o .  
> **Best probable translation:** we are smarter , more flexible , capable of learning more about , we survive in different environments , where we move people around the world and we traveled to space .  
> **Ground Truth:** we 're smarter , we 're more flexible , we can learn more , we survive in more different environments , we migrated to cover the world and even go to outer space .  
>
> **Translating from:** olhando para tr??s para os destro??os e desespero daqueles anos , parece-me agora como se algu??m tivesse morrido naquele lugar , e , no entanto , uma outra pessoa foi salva .  
> **Best probable translation:** looking behind the rubble and despair from those years , it seems to me now as someone had died in that place , and yet another person was saved .  
> **Ground Truth:** now looking back on the wreckage and despair of those years , it seems to me now as if someone died in that place , and yet , someone else was saved .  
>
> **Translating from:** o c??rebro pega em informa????o sem sentido e faz sentido a partir disso , o que significa que nunca vemos o que l?? est?? , nunca vemos informa????o , s?? vemos o que nos foi ??til ver no passado .  
> **Best probable translation:** the brain takes information without sense , and makes sense from it , which means that we never see what ' s there , we never see information , we see what was useful to see in the past .  
> **Ground Truth:** right ? the brain takes meaningless information and makes meaning out of it , which means we never see what 's there , we never see information , we only ever see what was useful to see in the past .  
>

#### Citation:

Please cite this work as:
```bibtex
@misc{gokden2021power,
      title={Power Law Graph Transformer for Machine Translation and Representation Learning}, 
      author={Burc Gokden},
      year={2021},
      eprint={2107.02039},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

