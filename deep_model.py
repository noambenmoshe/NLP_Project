import pandas as pd
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel,pipeline
from transformers import BertModel, BertTokenizerFast
from transformers import AutoConfig, AutoModelForMaskedLM, TrainingArguments,Trainer, DataCollatorForWholeWordMask
from datasets import Dataset
from sklearn.metrics import f1_score
from HeBERT.src.HebEMO import *

def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=2)
    labels = predictions.label_ids
    # f1 = f1_score(preds, labels, average='binary')
    print(f' np.sum(preds == labels) = {np.sum(preds == labels)}')
    print(f'np.size(labels) = {np.size(labels)}')
    acc = np.sum(preds == labels) / np.size(labels)
    # return {'f1': f1, 'acc': acc}
    return{'acc': acc}


def get_data():
    path_to_rbdf = '/MLAIM/AIMLab/Shany/databases/rbafdb/'
    path_to_dictionary = path_to_rbdf + 'documentation/reports/RBAF_reports.xlsx'
    path_to_validation = path_to_rbdf + 'documentation/reports/test_ann_sheina.xlsx'
    path_to_test = path_to_rbdf + 'documentation/reports/test_ann_sheina2.xlsx'
    my_dictionary = pd.read_excel(path_to_dictionary, engine='openpyxl')
    validation_set = pd.read_excel(path_to_validation, engine='openpyxl')
    test_set = pd.read_excel(path_to_test, engine='openpyxl')
    test_set_ids = test_set['Unnamed: 0'].tolist()
    validation_set_ids = validation_set['Unnamed: 0'].tolist()
    #remove test_set ids from the training set
    my_dictionary = my_dictionary[~my_dictionary['holter_id'].isin(test_set_ids)]
    my_dictionary = my_dictionary[~my_dictionary['holter_id'].isin(validation_set_ids)]

    return my_dictionary


def heBert_LM():
    # HebEMO_model = HebEMO()
    # hebEMO_df = HebEMO_model.hebemo(text='החיים יפים ומאושרים', plot=True)


    tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
    model = AutoModel.from_pretrained("avichr/heBERT")

    fill_mask = pipeline(
        "fill-mask",
        model="avichr/heBERT",
        tokenizer="avichr/heBERT"
    )
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    alephbert = BertModel.from_pretrained('onlplab/alephbert-base')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='onlplab/alephbert-base')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--prob_for_mask", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--out_dir", help="dir to save trained model", type=str, default='./Output')
    args = parser.parse_args()
    return args
def combine_text(df, col_a = 'text a', col_b ='text b'):
    combined_text_list = []
    for i, id in enumerate(df['holter_id']):
        j = int(df['holter_id'][df['holter_id'] == id].index.values)
        text_a = df[col_a][j]
        text_b = df[col_b][j]
        if pd.isna(text_a) or text_a == 0 or text_a == ' ':
            text = ''
        else:
            text = text_a
        if not (pd.isna(text_b) or text_b == 0 or text_b == ' '):
            text = text + text_b

        combined_text_list.append(text)
    assert len(combined_text_list) == len(
        df), f' len(combined_text_list)= {len(combined_text_list)} len(my_dictionary)={len(my_dictionary)} need to be same length'
    df['Text'] = combined_text_list
    df = df.drop(df[df.Text == ''].index)
    return df

def AlephBert_LM():
    args = get_args()

    path_to_rbdf = '/MLAIM/AIMLab/Shany/databases/rbafdb/'
    main_path = '/home/b.noam/NLP_Project/'
    path_to_dictionary = path_to_rbdf + 'documentation/reports/RBAF_reports.xlsx'
    path_to_validation = main_path + 'validation_set.xlsx'
    path_to_test = main_path + 'test_set.xlsx'

    my_dictionary = pd.read_excel(path_to_dictionary, engine='openpyxl')
    validation_set = pd.read_excel(path_to_validation, engine='openpyxl')
    test_set = pd.read_excel(path_to_test, engine='openpyxl')

    test_set_ids = test_set['holter_id'].tolist()
    validation_set_ids = validation_set['holter_id'].tolist()
    # remove test_set ids from the training set
    my_dictionary = my_dictionary[~my_dictionary['holter_id'].isin(test_set_ids)]
    # remove validation_set ids from the training set
    my_dictionary = my_dictionary[~my_dictionary['holter_id'].isin(validation_set_ids)]

    my_dictionary =combine_text(my_dictionary, col_a='טקסט מתוך סיכום', col_b='טקסט מתוך תוצאות')
    validation_set = combine_text(validation_set)

    dataset = {
        'train': Dataset.from_pandas(my_dictionary.astype(str)),
        'val': Dataset.from_pandas(validation_set.astype(str))
    }


    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_datasets = {'train': dataset['train'].map(tokenizer, input_columns=['Text'], fn_kwargs={"max_length": args.max_length, #Todo check that this is a good max
                                                                                        "truncation": True,
                                                                                        "padding": "max_length"}),
                          'val': dataset['val'].map(tokenizer, input_columns=['Text'],
                                               fn_kwargs={"max_length": args.max_length,
                                                          # Todo check that this is a good max
                                                          "truncation": True,
                                                          "padding": "max_length"})
                          }

    tokenized_datasets['train'].set_format('torch')
    tokenized_datasets['val'].set_format('torch')
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=args.prob_for_mask)

    # creat trainer
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

    # train
    trainer.train()
    print('done')

if __name__ == '__main__':
    AlephBert_LM()
