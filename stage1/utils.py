
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn import metrics

def convert_onehot(x, max=54):
    return np.eye(max)[x]

def row_wise_f1_score_micro_numpy(y_true, y_pred, threshold=0.5, count=5):
    """ 
    @author shonenkov 
    
    y_true - 2d npy vector with gt
    y_pred - 2d npy vector with prediction
    threshold - for round labels
    count - number of preds (used sorting by confidence)
    """
    def meth_agn_v2(x, threshold):
        idx, = np.where(x > threshold)
        return idx[np.argsort(x[idx])[::-1]]

    F1 = []
    for preds, trues in zip(y_pred, y_true):
        TP, FN, FP = 0, 0, 0
        preds = meth_agn_v2(preds, threshold)[:count]
        trues = meth_agn_v2(trues, threshold)
        for true in trues:
            if true in preds:
                TP += 1
            else:
                FN += 1
        for pred in preds:
            if pred not in trues:
                FP += 1
        F1.append(2*TP / (2*TP + FN + FP))
    return np.mean(F1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricMeter(object):

    def __init__(self, train=True):
        self.reset()
        self.train = train
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_true_ = []
        self.y_pred_ = []
        self.acc = 0
        self.f1 = 0
        self.map = 0
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist()) 
        self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())
        
    
    @property
    def avg(self):

        #self.y_true = np.concatenate(self.y_true, axis=0)
        #self.y_pred = np.concatenate(self.y_pred, axis=0)

        #print(self.y_true[0])
        #print(self.y_pred[0])

        if self.train:
            self.acc = metrics.accuracy_score(np.argmax(self.y_true, axis=1), np.argmax(self.y_pred, axis=1))
            self.auc = 0#metrics.roc_auc_score(self.y_true, self.y_pred, average=None)
            self.f1 = 0#metrics.f1_score(np.argmax(self.y_true, axis=1), np.argmax(self.y_true, axis=1), average=None)
            self.map = 0#np.mean(metrics.average_precision_score(self.y_true, self.y_pred)
            self.row_f1 = 0
        else:
            self.acc = metrics.accuracy_score(np.argmax(self.y_true, axis=1), np.argmax(self.y_pred, axis=1))
            self.auc = 0 #np.mean(metrics.roc_auc_score(self.y_true, self.y_pred, average=None))
            self.f1 = metrics.f1_score(np.argmax(self.y_true, axis=1), np.argmax(self.y_pred, axis=1), average='macro')
            self.map = np.nan_to_num(
                    np.mean(metrics.average_precision_score(self.y_true, self.y_pred, average=None))
                ).mean()
            
            self.row_f1 = row_wise_f1_score_micro_numpy(np.array(self.y_true), np.array(self.y_pred))

        return {
            "acc" : self.acc,
            "f1" : self.f1,
            "auc" : self.auc,
            "map" : self.map,
            "row_f1" : self.row_f1
        }