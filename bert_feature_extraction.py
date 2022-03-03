import torch
import argparse
import numpy as np
import os
from transformers import AutoTokenizer,TrainingArguments,Trainer, AutoModelForSequenceClassification
from datasets import Dataset
import utils
from preprocess import load_preprocessed
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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



def data_feature_extraction(model, config, train_set_l, val_set, label_col = 'AFIB'):
    col_name = 'Text' if config['language'] == 'HE' else 'text_en'
    dataset = {
        'train': Dataset.from_pandas(train_set_l),
        'val': Dataset.from_pandas(val_set.astype(str))
    }

    #convert label_col from str to int
    dataset['val'] = dataset['val'].map(lambda x: x.update({label_col: int(x[label_col])}) or x)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenized_datasets = {'train': dataset['train'].map(tokenizer, input_columns=[col_name],
                                                        fn_kwargs={"max_length": config['max_length'],
                                                                   # Todo check that this is a good max
                                                                   "truncation": True,
                                                                   "padding": "max_length"}),
                          'val': dataset['val'].map(tokenizer, input_columns=[col_name],
                                                    fn_kwargs={"max_length": config['max_length'],
                                                               "truncation": True,
                                                               "padding": "max_length"})
                          }

    tokenized_datasets['train'].set_format('torch')
    tokenized_datasets['val'].set_format('torch')

    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', dataset[split][label_col])

    training_args = TrainingArguments(output_dir=os.path.join(model_dir, 'feature_extraction/'),
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1,)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],)

    val_loader = trainer.get_eval_dataloader()
    all_labels = []
    all_features = []
    model.eval()
    for batch in val_loader:
        inp = {k: v.to(model.device) for k,v in batch.items()}
        features = bert_feature_extraction(model, **inp)
        all_labels.append(inp['labels'].detach().cpu())
        all_features.append(features.detach().cpu())

    all_labels = torch.concat(all_labels).numpy()
    all_features = torch.concat(all_features).numpy()

    return all_features, all_labels


def plot_tsne(X, y, model_name=''):

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
        ax.scatter(X_embedded[np.where(y == 0)][:, 0], X_embedded[np.where(y == 0)][:, 1], color='r', marker='*', label='False')
        ax.scatter(X_embedded[np.where(y == 1)][:, 0], X_embedded[np.where(y == 1)][:, 1], color='b', marker='x', label='Positive')
        ax.grid()
        ax.legend()
        ax.set_title("2D t-SNE - {}".format(model_name))

        plt.show()

    return X_embedded


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

def plot_pca(X, y, model_name=''):
    dim = 2
    perplexity = 30.0
    scale_data = True
    pca = PCA(n_components=dim)

    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_embedded = pca.fit_transform(X)
    if dim == 2:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X_embedded[np.where(y == 0)][:, 0], X_embedded[np.where(y == 0)][:, 1], color='r', marker='*', label='False')
        ax.scatter(X_embedded[np.where(y == 1)][:, 0], X_embedded[np.where(y == 1)][:, 1], color='b', marker='x', label='Positive')
        ax.grid()
        ax.legend()
        ax.set_title("2D PCA - {}".format(model_name))
        plt.show()
    return X_embedded


if __name__ == '__main__':
    df = load_preprocessed('/home/b.noam/NLP_Project/preprocessed_data/')
    train_set_nl, val_set, test_set, train_set_l = df['train_set'], df['val_set'], df['test_set'], df['labeled_data']
    # model_configs = [
    #     {'model_dir': }
    # ]
    models_dir = './models2/alephbert/'
    label_col = 'AFIB'
    bert_model = 'AlephBert'



    # else:
    #     model = AutoModelForSequenceClassification.from_pretrained(model_name,
    #                                                                num_labels=2)
    #     config = {}
    x_pca_all = []
    x_tsne_all = []
    y_all = []
    names_all = []
    # bluebert LM
    model_dir = os.path.join(models_dir, 'lm')
    model_name = bert_model + '-LM'
    model, config = utils.load_model_for_classificaion(os.path.join(model_dir, 'best_model/'))
    all_features, all_labels = data_feature_extraction(model, config, train_set_l, test_set, label_col=label_col)
    x_tsne_all.append(plot_tsne(all_features, all_labels, model_name=model_name))
    x_pca_all.append(plot_pca(all_features, all_labels, model_name=model_name))
    y_all.append(all_labels)
    names_all.append(model_name)

    # bluebert pretrained - use config of bluebert_lm
    model_name = bert_model + '-Pretrained'
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=config['num_labels'])
    all_features, all_labels = data_feature_extraction(model, config, train_set_l, test_set, label_col=label_col)
    x_tsne_all.append(plot_tsne(all_features, all_labels, model_name=model_name))
    x_pca_all.append(plot_pca(all_features, all_labels, model_name=model_name))
    y_all.append(all_labels)
    names_all.append(model_name)

    # bluebert LM AFIB
    model_name = bert_model + '-LM-Class'
    model_dir = os.path.join(models_dir, 'lm_afib')
    model, config = utils.load_model_for_classificaion(os.path.join(model_dir, 'best_model/'))
    all_features, all_labels = data_feature_extraction(model, config, train_set_l, test_set, label_col=label_col)
    x_tsne_all.append(plot_tsne(all_features, all_labels, model_name=model_name))
    x_pca_all.append(plot_pca(all_features, all_labels, model_name=model_name))
    y_all.append(all_labels)
    names_all.append(model_name)

    # bluebert AFIB
    model_name = bert_model + '-Class'
    model_dir = os.path.join(models_dir, 'afib')
    model, config = utils.load_model_for_classificaion(os.path.join(model_dir, 'best_model/'))
    all_features, all_labels = data_feature_extraction(model, config, train_set_l, test_set, label_col=label_col)
    x_tsne_all.append(plot_tsne(all_features, all_labels, model_name=model_name))
    x_pca_all.append(plot_pca(all_features, all_labels, model_name=model_name))
    y_all.append(all_labels)
    names_all.append(model_name)

    # plot all tsne
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    for i, (X_embedded, y, name) in enumerate(zip(x_tsne_all, y_all, names_all)):

        ax[int(i / 2), i % 2].scatter(X_embedded[np.where(y == 0)][:, 0], X_embedded[np.where(y == 0)][:, 1], color='r',
                                      marker='*',
                                      label='Non AF')
        ax[int(i / 2), i % 2].scatter(X_embedded[np.where(y == 1)][:, 0], X_embedded[np.where(y == 1)][:, 1], color='b',
                                      marker='x',
                                      label='AF')
        ax[int(i / 2), i % 2].legend()
        ax[int(i / 2), i % 2].set_title(name)
    ax[0,1].legend()
    fig.suptitle('2D t-SNE')
    plt.savefig(os.path.join(models_dir, bert_model+'_pca.png'))
    plt.show()

    # plot all pca
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    for i, (X_embedded, y, name) in enumerate(zip(x_pca_all, y_all, names_all)):
        ax[int(i/2),i%2].scatter(X_embedded[np.where(y == 0)][:, 0], X_embedded[np.where(y == 0)][:, 1], color='r', marker='*',
                   label='Non AF')
        ax[int(i/2),i%2].scatter(X_embedded[np.where(y == 1)][:, 0], X_embedded[np.where(y == 1)][:, 1], color='b', marker='x',
                   label='AF')
        ax[int(i/2),i%2].legend()
        ax[int(i/2),i%2].set_title(name)
    ax[0, 1].legend()
    fig.suptitle('2D PCA')
    plt.savefig(os.path.join(models_dir, bert_model+'_pca.png'))
    plt.show()
