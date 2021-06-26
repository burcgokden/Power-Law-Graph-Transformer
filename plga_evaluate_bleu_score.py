
import os
import common as cm
import sacrebleu
import nmt_data_prep as nmtdp
import plga_transformer_run_model as plga_run



def evaluate_bleu(model_dict,
                  model_name,
                  model_type,
                  src_lang='pt',
                  tgt_lang='en',
                  dataset_file='ted_hrlr_translate/pt_to_en',
                  revert_order='False',
                  inp_obj=None,
                  chkpt_path="./model_saves/",
                  data_path="./model_data/",
                  load_ckpt='train',
                  tok_model_name="./ted_hrlr_translate_pt_en_tokenizer",
                  max_length=50,
                  ds_max_length=None,
                  verbose=False):
    '''
    Evaluate bleu with transformer model from dataset using greedy search.
    Args:
        model_dict: model parameters for transformer.
        model_name: model name
        model_type: "train" or "val"
        src_lang: source language abbreviation as string
        tgt_lang: target language abbreviation as string
        dataset_file: path to tensorflow dataset
        revert_order: If True, it reverts the order of language pairs in dataset_file. Reverted order should match
                      src_lang/tgt_lang assignment.
        inp_obj: dataset object if it was already created.
        bleu_ref_filepath: file path for bleu references to be loaded if exists or to be saved.
        chkpt_path: path where model checkpoints can be loaded from.
        data_path: path to save model date or load from
        load_ckpt:'train' or 'val' checkpoint to load from.
        tok_model_name: file path for tokenizer model.
        max_length: Offset for Maximum iteration to complete for predicting tokens.
        ds_max_length: Maximum token length for filtering sentences in dataset. Set to None for no filtering.
        verbose: If True print out more details.
    '''

    if inp_obj is None:
        print("Getting Inputs")
        inp_obj=nmtdp.src_tgt_data_prep(src_lang=src_lang,
                                             tgt_lang=tgt_lang,
                                             BUFFER_SIZE=20000,
                                             BATCH_SIZE = 64,
                                             dataset_file=dataset_file,
                                             load_dataset=True,
                                             train_percent=None,
                                             model_name = tok_model_name,
                                             revert_order=revert_order,
                                             shuffle_set=True,
                                             shuffle_files=True,
                                             MAX_LENGTH=ds_max_length,
                                             verbose=verbose)

    print("Initializing model e2e object")
    e2e_obje = plga_run.plga_transformer_e2e(inp_obj.tokenizers_src, inp_obj.tokenizers_tgt,
                                      checkpoint_path=chkpt_path,
                                      hpdict=model_dict,
                                      load_ckpt=load_ckpt
                                      )

    test_inputs, test_refs=[],[]
    for src, tgt in inp_obj.test_examples:
        test_inputs.append(src.numpy().decode('utf-8'))
        test_refs.append(tgt.numpy().decode('utf-8'))

    dsmaxlen= ds_max_length if ds_max_length is not None else "_unpadded"
    model_name1 = f"{model_type}_{model_name}_greedy_evallen{max_length}_dslen{dsmaxlen}"
    model_pred_path=os.path.join(data_path,f"predictions_{model_name1}.pkl")
    if not os.path.exists(model_pred_path):
        print("Predicting test sentences")
        pred_sent_lst, pred_tok_lst=e2e_obje.evaluate_test(test_inputs, test_refs,
                                                            max_length=max_length,
                                                            filename=model_pred_path, verbose=verbose)
    else:
        print("Loading predictions from file", model_pred_path)
        pred_sent_lst, pred_tok_lst=cm.pklload(model_pred_path)

    bleu_ref_filepath = os.path.join(data_path, f"ref_bleu_test_{model_name1}.pkl")
    if not os.path.exists(bleu_ref_filepath):
        print("saving reference test sentences for target")
        corpus_test_refs = [test_refs]
        cm.pklsave(bleu_ref_filepath, corpus_test_refs)
    else:
        corpus_test_refs = cm.pklload(bleu_ref_filepath)

    model_bleu_score = sacrebleu.corpus_bleu(pred_sent_lst, corpus_test_refs)

    print("corpus bleu score:", model_bleu_score)

    return model_bleu_score

def beam_evaluate_bleu(model_dict,
                       beam_size,
                       model_name,
                       model_type,
                       src_lang='pt',
                       tgt_lang='en',
                       dataset_file='ted_hrlr_translate/pt_to_en',
                       revert_order='False',
                       inp_obj=None,
                       chkpt_path="./model_saves/",
                       data_path="./model_data/",
                       load_ckpt='train',
                       tok_model_name="./ted_hrlr_translate_pt_en_tokenizer",
                       max_length=50,
                       ds_max_length=None,
                       verbose=False):
    '''
    Evaluate bleu with transformer model from dataset using beam search.
    Args:
        model_dict: model parameters for transformer.
        beam_size: beam length for beam search.
        model_name: model name
        model_type: "train" or "val"
        src_lang: source language abbreviation as string
        tgt_lang: target language abbreviation as string
        dataset_file: path to tensorflow dataset
        revert_order: If True, it reverts the order of language pairs in dataset_file. Reverted order should match
                      src_lang/tgt_lang assignment.
        inp_obj: dataset object if it was already created.
        bleu_ref_filepath: file path for bleu references to be loaded if exists or to be saved.
        chkpt_path: path where model checkpoints can be loaded from.
        data_path: path to save model date or load from
        load_ckpt:'train' or 'val' checkpoint to load from.
        tok_model_name: file path for tokenizer model.
        max_length: Offset for Maximum iteration to complete for predicting tokens.
        ds_max_length: Maximum token length for filtering sentences in dataset.
        smoothing_function: Smoothing method for bleu score. Default is None (smooth.method0()).
        verbose: If True print out more details.
    Returns bleu score and a list of sentences not predicted if any.
    Saves the predicted sentences and tokens in a file.
    '''

    if inp_obj is None:
        print("Getting Inputs")
        inp_obj=nmtdp.src_tgt_data_prep(src_lang=src_lang,
                                             tgt_lang=tgt_lang,
                                             BUFFER_SIZE=20000,
                                             BATCH_SIZE = 64,
                                             dataset_file=dataset_file,
                                             load_dataset=True,
                                             train_percent=None,
                                             model_name = tok_model_name,
                                             revert_order=revert_order,
                                             shuffle_set=True,
                                             shuffle_files=False,
                                             MAX_LENGTH=ds_max_length,
                                             verbose=verbose)

    print("Initializing model e2e object")
    e2e_obje = plga_run.plga_transformer_e2e(inp_obj.tokenizers_src, inp_obj.tokenizers_tgt,
                                      checkpoint_path=chkpt_path,
                                      hpdict=model_dict,
                                      load_ckpt=load_ckpt
                                      )

    test_inputs, test_refs=[],[]
    for src, tgt in inp_obj.test_examples:
        test_inputs.append(src.numpy().decode('utf-8'))
        test_refs.append(tgt.numpy().decode('utf-8'))

    dsmaxlen= ds_max_length if ds_max_length is not None else "_unpadded"
    model_name1 = f"{model_type}_{model_name}_beamsize{beam_size}_evallen{max_length}_dslen{dsmaxlen}"
    model_pred_path=os.path.join(data_path,f"predictions_{model_name1}.pkl")
    if not os.path.exists(model_pred_path):
        print("Predicting test sentences")
        pred_sent_lst_fullbeam, pred_tok_lst, final_beam_seq, unpreds=e2e_obje.beam_evaluate_test(test_inputs, test_refs, beam_size=beam_size,
                                                               max_length=max_length, filename=model_pred_path, verbose=verbose)
    else:
        print("Loading predictions from file", model_pred_path)
        pred_sent_lst_fullbeam, pred_tok_lst, final_beam_seq, unpreds=cm.pklload(model_pred_path)

    #get the highest probable sentence in pred_sent_lst for bleu evaluation
    pred_sent_lst=[]
    k=0
    for i, pred_sent_beam in enumerate(pred_sent_lst_fullbeam):
        if len(pred_sent_beam) > 0:
            pred_sent_lst.append(pred_sent_beam[0])
        else:
            print(f"{i+1}th test sentence was empty: {unpreds[k]}")
            print("")
            k+=1

    bleu_ref_filepath = os.path.join(data_path, f"ref_bleu_test_{model_name1}.pkl")
    if not os.path.exists(bleu_ref_filepath):
        print("saving reference test sentences for target")
        corpus_test_refs = [test_refs]
        cm.pklsave(bleu_ref_filepath, corpus_test_refs)
    else:
        corpus_test_refs = cm.pklload(bleu_ref_filepath)

    model_bleu_score = sacrebleu.corpus_bleu(pred_sent_lst, corpus_test_refs)

    print("corpus bleu score:", model_bleu_score)

    return model_bleu_score, unpreds