import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


def train(model, train_dataloader, optimizer, criterion, epoch, device):
    # For computing average of loss, metric, accuracy
    loss_list = []
    metric_list = []
    acc_list = []
    
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
        x, target = data.to(device), target.to(device)

        # Forward
        y_pred = model(x)

        # Computing Loss
        train_loss, train_metric = criterion.loss_fn(target, y_pred)
        
        max_idx = torch.argmax(y_pred, dim=-1)
        loss_list.append(sum(train_loss.cpu().numpy())//len(data))
        metric_list.append(sum(train_metric.cpu().numpy())//len(data))
        acc = sum(target.cpu().numpy()==max_idx.cpu().numpy())//len(data)
        acc_list.append(acc)

        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    # TODO: Save in logfile 
    avg_acc = sum(acc_list)
    avg_metric = sum(metric_list)
    avg_loss = sum(loss_list)
    print(f"[Epoch {epoch}] Train Accuracy: {avg_acc}\t F1 Score {avg_metric}\t Loss: {avg_loss}")
    torch.save(model, f'./checkpoints/model_epoch{epoch}.pt')