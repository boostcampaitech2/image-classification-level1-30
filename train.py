import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, train_dataloader, optimizer, criterion, epoch, device, min_loss):
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
        loss_list.append(train_loss.detach().cpu().numpy())
        metric_list.append(train_metric)
        acc = sum(target.detach().cpu().numpy()==max_idx.detach().cpu().numpy())/len(data)
        acc_list.append(acc)

        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    # TODO: Save in logfile
    avg_acc = sum(acc_list)/len(train_dataloader)
    avg_metric = sum(metric_list)/len(train_dataloader)
    avg_loss = sum(loss_list)/len(train_dataloader)
    print(f"[Epoch {epoch}] Train Accuracy: {avg_acc}\t F1 Score: {avg_metric}\t Loss: {avg_loss}")
    if min_loss > avg_loss:
        torch.save(model, f'./checkpoints/model.pt')
    return min(min_loss, avg_loss)