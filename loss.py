import torch
import torch.nn as nn

from sklearn.metrics import f1_score


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def f1_loss(self, y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)
                
        f1 = f1_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')

        return f1

    def loss_fn(self, y: torch.Tensor, y_pred: torch.Tensor):
        train_metric = self.f1_loss(y, y_pred)
        train_loss = self.loss(y_pred, y)
        return train_loss, train_metric