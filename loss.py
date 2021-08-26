import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion():
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()

    # Metric
    def f1_loss(self, y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
        '''Calculate F1 score. Can work with gpu tensors
        
        The original implmentation is written by Michal Haltuf on Kaggle.
        
        Returns
        -------
        torch.Tensor
            `ndim` == 1. 0 <= val <= 1
        
        Reference
        ---------
        - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
        
        '''
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2
        

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)
            
        
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        
        epsilon = 1e-7
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        f1 = 2* (precision*recall) / (precision + recall + epsilon)
        f1.requires_grad = is_training
        return f1


    def loss_fn(self, y, y_pred):
        train_metric = self.f1_loss(y, y_pred)
        train_loss = self.loss(y_pred, y)
        return train_loss, train_metric
