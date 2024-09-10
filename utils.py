import dgl
import torch
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

from dataset import Dataset

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def train_test_val(labels, train_size, seed=0):
    index = list(range(len(labels)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=train_size,
                                                            random_state=seed, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=seed, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    return train_mask, val_mask, test_mask 


# def load_data(name, homo=True):
#     graph = load_graphs('datasets/'+name)[0][0]
#     return graph


def load_data(name, homo=True):
    if name == 'amazon' or name == 'yelp':
        g = Dataset(name, homo).graph
        return g
    else:
        graph = load_graphs('datasets/'+name)[0][0]
    return graph

# def load_data(name, homo=True):
#     if name == 'yelp':
#         dataset = FraudYelpDataset()
#         graph = dataset[0]
#         graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
#         graph = dgl.add_self_loop(graph)
#     graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
#     graph.ndata['feature'] = graph.ndata['feature'].float()
#     return graph



# if __name__ == "__main__":
#     labels = torch.tensor([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
#     train_mask, val_mask, test_mask = train_test_val(labels, 0.6)
#     print(train_mask, val_mask, test_mask)

