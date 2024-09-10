
import torch
import numpy as np
import dgl
import argparse
import random
import os

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score

from tqdm import tqdm
from utils import load_data, train_test_val, get_best_f1
from Auto_AD import *
import torch.nn as nn

import warnings

parser = argparse.ArgumentParser(description='Auto-AD')
parser.add_argument('--dataset', type=str, default='yelp')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=1999)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--epochs', type=int, default =500)
parser.add_argument('--train_ratio', type=float, default=0.01)
parser.add_argument('--hidden_dim', type=int, default=32)

def seed_everything(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":


    args = parser.parse_args()

    seed_everything(args.seed)
    data = load_data(args.dataset)

    features = data.ndata['feature']

    labels = data.ndata['label']
    train_mask, val_mask, test_mask = train_test_val(labels, args.train_ratio)
    
    features = features.to(args.device)
    data = data.to(args.device)
    model = Auto_AD(features.shape[1], args.hidden_dim, 2, data, 2, False)
    model = model.to(args.device)
    labels = labels.to(args.device)
    labels_cpu = labels.to('cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    with tqdm(total=args.epochs) as pbar:
        pbar.set_description('training')
        for epoch in range(args.epochs):
            logit = model(features)
            loss = loss_fn(logit[train_mask], labels[train_mask]) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().numpy()

            probs = logit.softmax(1)[:, 1]
            probs = probs.cpu().detach()
            f1, thres = get_best_f1(labels_cpu[test_mask], probs[test_mask])
            preds = np.zeros_like(labels_cpu) 
            preds[probs[:] > thres] = 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_auc = roc_auc_score(labels_cpu[test_mask], probs[test_mask])
                val_rec = recall_score(labels_cpu[test_mask], preds[test_mask])
                val_pre = precision_score(labels_cpu[test_mask], preds[test_mask])
                val_mf1 = f1_score(labels_cpu[test_mask], preds[test_mask], average='macro')

            pbar.set_postfix(loss="{:.3f}".format(loss.item()), auc="{:.3f}".format(val_auc), rec="{:.3f}".format(val_rec), pre="{:.3f}".format(val_pre), mf1="{:.3f}".format(val_mf1)) 
            pbar.update(1)     

    model.eval()
    with torch.no_grad():
        logit = model(features)
        logit = logit.cpu().detach() 
        probs = logit.softmax(1)[:, 1] 
        probs = probs.cpu().detach()
        f1, thres = get_best_f1(labels_cpu[val_mask], probs[val_mask])
        preds = np.zeros_like(labels_cpu)
        preds[probs[:] > thres] = 1
        auc = roc_auc_score(labels_cpu[test_mask], probs[test_mask])
        rec = recall_score(labels_cpu[test_mask], preds[test_mask])
        pre = precision_score(labels_cpu[test_mask], preds[test_mask])
        mf1 = f1_score(labels_cpu[test_mask], preds[test_mask], average='macro')
    print('auc      :{:.4}'.format(auc))
    print('rec-macro:{:.4}'.format(rec))
    print('pre      :{:.4}'.format(pre))
    print('f1-macro :{:.4}'.format(mf1))