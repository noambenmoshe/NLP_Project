import pandas as pd
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel,pipeline
from transformers import BertModel, BertTokenizerFast
from transformers import AutoConfig, AutoModelForMaskedLM, TrainingArguments,Trainer
from datasets import Dataset
from sklearn.metrics import f1_score
from HeBERT.src.HebEMO import *

def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    f1 = f1_score(preds, labels, average='binary')
    acc = np.sum(preds == labels) / np.size(labels)
    return {'f1': f1, 'acc': acc}


def get_data():
    path_to_rbdf = '/MLAIM/AIMLab/Shany/databases/rbafdb/'
    path_to_dictionary = path_to_rbdf + 'documentation/reports/RBAF_reports.xlsx'
    my_dictionary = pd.read_excel(path_to_dictionary, engine='openpyxl')
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

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--out_dir", help="dir to save trained model", type=str, default='./Output')
    args = parser.parse_args()
    return args

def AlephBert_LM():
    args = get_args()
    path_to_rbdf = '/MLAIM/AIMLab/Shany/databases/rbafdb/'
    path_to_dictionary = path_to_rbdf + 'documentation/reports/RBAF_reports.xlsx'
    my_dictionary = pd.read_excel(path_to_dictionary, engine='openpyxl')
    data_pd = my_dictionary

    dataset = Dataset.from_pandas(data_pd.astype(str))
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = dataset.map(tokenizer, input_columns=['טקסט מתוך סיכום', 'טקסט מתוך תוצאות'], fn_kwargs={"max_length": args.max_length, #Todo check that this is a good max
                                                                                        "truncation": True,
                                                                                        "padding": "max_length"})
    tokenized_datasets.set_format('torch')
    # creat trainer
    training_args = TrainingArguments(output_dir=args.out_dir, overwrite_output_dir=True,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      gradient_accumulation_steps=args.grad_accum,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='acc',
                                      greater_is_better=True, evaluation_strategy='epoch', do_train=True,
                                      num_train_epochs=args.epochs,
                                      report_to= None,
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        # eval_dataset=tokenized_datasets['test'],
        compute_metrics=metric_fn
    )

    # train
    trainer.train()
    print('done')

if __name__ == '__main__':
    AlephBert_LM()