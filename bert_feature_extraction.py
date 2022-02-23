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
from HeBERT.src.HebEMO import *
from deep_model import get_args

def bert_feature_extraction(model,
                            input_ids = None,
                            attention_mask = None,
                            token_type_ids = None,
                            position_ids = None,
                            head_mask = None,
                            inputs_embeds = None,
                            labels = None,
                            output_attentions = None,
                            output_hidden_states = None,
                            return_dict = None,

                        ):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    outputs = model.bert(
    input_ids,
    attention_mask = attention_mask,
    token_type_ids = token_type_ids,
    position_ids = position_ids,
    head_mask = head_mask,
    inputs_embeds = inputs_embeds,
    output_attentions = output_attentions,
    output_hidden_states = output_hidden_states,
    return_dict = return_dict,
    )

    pooled_output = outputs[1]

    output = model.dropout(pooled_output)
    return output



def data_feature_extraction(args, train_set_l, val_set, label_col = 'AFIB'):

    dir_to_load = args.out_dir_Classify
    # dir_to_load = args.out_dir_LM

    model, config = utils.load_model_for_classificaion(os.path.join(dir_to_load, 'best_model/'))
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    if config['language'] == 'HE':
        col_name = 'Text'
    else:
        col_name ='text_en'


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


    trainer = Trainer(
        model=model,
        # args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        # compute_metrics=metric_fn_clssify
    )

    val_loader = trainer.get_eval_dataloader()

    all_labels = []
    all_features = []
    for batch in val_loader:
        inp = {k: v.to(model.device) for k,v in batch.items()}
        features = bert_feature_extraction(model, **inp)
        all_labels.append(inp['labels'].detach().cpu())
        all_features.append(features.detach().cpu())

    all_labels = torch.concat(all_labels).numpy()
    all_features = torch.concat(all_features).numpy()
    plot_tsne(all_features, all_labels)

def plot_tsne(X, y):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    dim = 2
    perplexity = 30.0
    scale_data = True
    t_sne = TSNE(n_components=dim, perplexity=perplexity)

    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_embedded = t_sne.fit_transform(X)
    if dim == 2:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X_embedded[np.where(y == 0)][:, 0], X_embedded[np.where(y == 0)][:, 1], color='r', marker='*', label='Malignant')
        ax.scatter(X_embedded[np.where(y == 1)][:, 0], X_embedded[np.where(y == 1)][:, 1], color='b', marker='x', label='Benign')
        ax.grid()
        ax.legend()
        ax.set_title("2D t-SNE of Bert hidden features")
        plt.show()

    print("done")
    # outlayer
# old/_t_sne.py:790: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
#   warnings.warn(
# np.max(X_embedded[np.where(y == 0)][:, 0])
# 5.112461
# np.argmax(X_embedded[np.where(y == 0)][:, 0])
# 26
# X_embedded[:,0] == 5.112461
# array([False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False,  True, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False])
# np.argmax(X_embedded[:,0] == 5.112461)
# 33



if __name__ == '__main__':
    args = get_args()
    df = load_preprocessed()
    train_set_nl, val_set, test_set, train_set_l = df['train_set'], df['val_set'], df['test_set'], df['labeled_data']
    data_feature_extraction(args, train_set_l, val_set)