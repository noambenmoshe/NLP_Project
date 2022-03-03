import pandas
import pandas as pd
import torch
import argparse
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel,pipeline
from transformers import BertModel, BertTokenizerFast
from transformers import AutoConfig, AutoModelForMaskedLM, TrainingArguments,Trainer, DataCollatorForWholeWordMask
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import utils
from preprocess import load_preprocessed
from sklearn.metrics import f1_score, confusion_matrix
from custom_trainer import CustomTrainer


def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=2)
    labels = predictions.label_ids
    # f1 = f1_score(preds, labels, average='binary')
    print(f' np.sum(preds == labels) = {np.sum(preds == labels)}')
    print(f'np.size(labels) = {np.size(labels)}')
    num_masks = np.sum(labels != -100)
    print(f'num_masks = {num_masks}')
    acc = np.sum(preds == labels) / num_masks
    # acc = np.sum(preds == labels) / np.size(labels)
    # return {'f1': f1, 'acc': acc}
    return{'acc': acc}


def metric_fn_clssify(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    f1 = f1_score(preds, labels, average='binary')
    acc = np.sum(preds == labels) / np.size(labels)
    cm1 = confusion_matrix(labels, preds)
    print('Confusion Matrix : \n', cm1)
    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Sensitivity : ', sensitivity1)
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Specificity : ', specificity1)
    print('f1 : ', f1)
    print('acc :', acc)

    return {'f1': f1, 'acc': acc, 'Sensitivity': sensitivity1, 'Specificity': specificity1 }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='onlplab/alephbert-base')  #bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12 #microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
    parser.add_argument("--language", type=str, default='HE')  #EN
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--prob_for_mask", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=400)
    # parser.add_argument("--out_dir_LM", help="dir to save trained model", type=str, default=None)
    # parser.add_argument("--out_dir_Classify", help="dir to save trained model", type=str, default=None)
    parser.add_argument("--load_dir", help="dir to load trained model [used for train classification with trained LM model]",
                        type=str, default=None)
    parser.add_argument("--out_dir", help="dir to save trained model [used to save LM / Classification trained models]",
                        type=str, default=None)
    parser.add_argument("--verbose", help="add extra prints and figures", type=bool, default=False)
    parser.add_argument("--weights", help="weights for cross entropy loss ", type=list, default=[1.0, 10.0])
    parser.add_argument("--data_dir", help="path to preprocessed data", type=str,
                        default='/home/b.noam/NLP_Project/preprocessed_data/')
    parser.add_argument("--task", type=str, choices=['LM', 'AFIB', 'VT'])
    args = parser.parse_args()
    return args


def LM_head(args, train_set, validation_set):
    print(f'args.lanuage == {args.language}')
    col_name = 'Text' if args.language == 'HE' else 'text_en'

    dataset = {
        'train': Dataset.from_pandas(train_set.astype(str)),
        'val': Dataset.from_pandas(validation_set.astype(str))
    }

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_datasets = {'train': dataset['train'].map(tokenizer, input_columns=[col_name],
                                                        fn_kwargs={"max_length": args.max_length, #Todo check that this is a good max
                                                                        "truncation": True,
                                                                        "padding": "max_length"}),

                          'val': dataset['val'].map(tokenizer, input_columns=[col_name],
                                               fn_kwargs={"max_length": args.max_length,
                                                          "truncation": True,
                                                          "padding": "max_length"})
                          }

    if args.verbose:
        utils.data_stats(tokenized_data=tokenized_datasets, out_dir=args.out_dir_LM, col_name=col_name)

    tokenized_datasets['train'].set_format('torch')
    tokenized_datasets['val'].set_format('torch')
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=args.prob_for_mask)

    # create trainer
    training_args = TrainingArguments(output_dir=args.out_dir, overwrite_output_dir=True,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      gradient_accumulation_steps=args.grad_accum,
                                      eval_accumulation_steps = args.grad_accum,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='acc',
                                      greater_is_better=True,
                                      evaluation_strategy='epoch',
                                      do_train=True,
                                      num_train_epochs=args.epochs,
                                      report_to= None,
                                      logging_steps=10
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        compute_metrics=metric_fn,
        data_collator=data_collator
    )
    if args.language == 'HE':
        dummy_test(model, tokenizer)
    # train
    trainer.train()
    if args.language == 'HE':
        dummy_test(model, tokenizer)
    # save best model
    save_dir = os.path.join(args.out_dir, 'best_model/')
    print("save best model in: ", save_dir)
    utils.save_model(model, vars(args), save_dir)
    print('done')

def classify_head(args, train_set_l, val_set, label_col='AFIB', mode='Train', load_dir=None):
    col_name = 'Text' if args.language == 'HE' else 'text_en'

    if load_dir is not None:
        print("[classify_head] load model From: {}".format(load_dir))
        model, _ = utils.load_model_for_classificaion(os.path.join(load_dir, 'best_model/'))
    else:
        print("[classify_head] start training from pretrained: {} ".format(args.model_name))
        # config = AutoConfig.from_pretrained(args.model_name)
        # config.max_position_embeddings = 1024
        # model = AutoModelForSequenceClassification.from_config(config)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    dataset = {
        'train': Dataset.from_pandas(train_set_l),
        'val': Dataset.from_pandas(val_set.astype(str))
    }

    #convert label_col from str to int
    dataset['val'] = dataset['val'].map(lambda x: x.update({label_col: int(x[label_col])}) or x)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = {'train': dataset['train'].map(tokenizer, input_columns=[col_name],
                                                        fn_kwargs={"max_length": args.max_length,
                                                                   # Todo check that this is a good max
                                                                   "truncation": True,
                                                                   "padding": "max_length"}),
                          'val': dataset['val'].map(tokenizer, input_columns=[col_name],
                                                    fn_kwargs={"max_length": args.max_length,
                                                               "truncation": True,
                                                               "padding": "max_length"})
                          }
    tokenized_datasets['train'].set_format('torch')
    tokenized_datasets['val'].set_format('torch')
    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', dataset[split][label_col])

    training_args = TrainingArguments(output_dir=args.out_dir, overwrite_output_dir=True,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      gradient_accumulation_steps=args.grad_accum,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='f1',
                                      greater_is_better=True, evaluation_strategy='epoch', do_train=True if mode == 'Train' else False,
                                      num_train_epochs=args.epochs,
                                      report_to= None,
                                      )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        compute_metrics=metric_fn_clssify,
        weights = args.weights
    )
    if mode == 'Train':
        # train
        trainer.train()
    elif mode == 'Eval':
        trainer.evaluate()
    # save best model
    save_dir = os.path.join(args.out_dir, 'best_model/')
    print("save best model in: ", save_dir)
    utils.save_model(model, vars(args), save_dir)
    print('done')



def dummy_test(model, tokenizer):
    print("dummy test")
    device = model.device.index if str(model.device).startswith('cuda') else -1
    nlp_fill_mask_ita = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    sentences =['הפעימות החדריות המוקדמות מקורן מספר [MASK] שונים',
                'הפעימות החדריות [MASK] מקורן מספר מוקדים שונים',
                 'רישום למשך כ- 48 [MASK], מקצב בסיסי סינוס 60-90 לקדה.',
                'רישום למשך כ- 48 שעות, מקצב בסיסי [MASK] 60-90 לקדה.',
                'מקצב סינוס עם הולכה עלייתית-חדרית עם מקטע [MASK] מוארך.',
                'מקצב סינוס עם הולכה עלייתית-[MASK] עם מקטע PR מוארך.',
                'מקצב סינוס עם הולכה [MASK]-חדרית עם מקטע PR מוארך.'
                ]
    for sentence in sentences:
        print("masked sentence:")
        print(sentence)
        res = nlp_fill_mask_ita(sentence)
        print("model output:")
        for _res in res:
            print(_res['sequence'])


if __name__ == '__main__':
    args = get_args()
    print("args: ", args)
    utils.set_seed(args.seed)
    # train_set_nl, val_set, test_set, train_set_l = preprocessing(args)
    df = load_preprocessed(args.data_dir)
    train_set_nl, val_set, test_set, train_set_l = df['train_set'], df['val_set'], df['test_set'], df['labeled_data']

    if args.task == 'LM':
        lm_train_set = pandas.concat([train_set_l, train_set_nl])
        print("start LM train")
        LM_head(args=args, train_set=lm_train_set, validation_set=val_set)

    else:
        label_col = args.task # VT of AFIB
        print("start Classification [{}] train".format(args.task))
        classify_head(args, train_set_l, val_set, label_col, mode='Train', load_dir=args.load_dir)

        print("start Classification [{}] validation".format(args.task))
        classify_head(args, train_set_l, val_set, label_col, mode='Eval', load_dir=args.out_dir)
        print("start Classification [{}] test".format(args.task))
        classify_head(args, train_set_l, test_set, label_col, mode = 'Eval', load_dir=args.out_dir)
