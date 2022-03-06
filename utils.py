import random
import numpy as np
import torch
import os
from pathlib import Path
import json
from transformers import AutoModelForSequenceClassification
import matplotlib.pyplot as plt




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, config, dir):
    '''
    saves model state_dict and config dictionary for reloading
    :param model: model to save
    :param config: moedl configuration params
    :param dir: where to save the model
    :return: None
    '''
    Path(dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.to('cpu').state_dict(), os.path.join(dir, 'model.pt'))
    with open(os.path.join(dir, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)

def load_model_for_classificaion(dir):
    with open(os.path.join(dir,'config.json'), "r") as fp:
        config = json.load(fp)

    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'],
                                                                                  num_labels=config['num_labels'])
    model.load_state_dict(torch.load(os.path.join(dir, 'model.pt')),strict=False)
    return model, config

def data_stats(tokenized_data, out_dir, col_name):
    # generate dataset statistics
    train_len = []
    val_len = []
    data_len  =[]

    for i in tokenized_data['train']:
        data_len.append(len(i[col_name]))

    for i in tokenized_data['val']:
        data_len.append(len(i[col_name]))


    plt.figure()
    hist = plt.hist(data_len,bins=1000, cumulative=True, label='CDF',
             histtype='step', alpha=0.8, color='k', density=True)
    plt.title('Training Set Descriptions Length CDF')
    plt.xlim(xmin=0, xmax=500)
    plt.savefig(os.path.join(out_dir, 'train_review_length.png'))
    plt.show()

    plt.figure()
    plt.hist(val_len, bins=1000, cumulative=True, label='CDF',
             histtype='step', alpha=0.8, color='k', density=True)
    plt.title('validation reviews lengths CDF')
    plt.xlim(xmin=0, xmax=500)

    plt.savefig(os.path.join(out_dir, 'val_review_length.png'))
    plt.show()

    return

def len_list(tokenized_data,col_name):
    data_len = []
    for i in tokenized_data['train']:
        data_len.append(len(i[col_name]))

    for i in tokenized_data['val']:
        data_len.append(len(i[col_name]))
    return data_len



