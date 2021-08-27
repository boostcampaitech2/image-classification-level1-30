import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset

import torch
import torch.nn as nn
import torch.nn.functional as F


def getKfoldDataloaders(dataset, batch_size, num_workers, train_idx, val_idx):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=val_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader


def ktrain(model, train_dataloader, val_loader, optimizer, criterion, epoch, device, min_loss, min_val_loss, fold):
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
        metric_list.append(train_metric.detach().cpu().numpy())
        acc = sum(target.detach().cpu().numpy()==max_idx.detach().cpu().numpy())/len(data)
        acc_list.append(acc)

        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    # validation
    with torch.no_grad():
        val_loss_list = []
        val_metric_list = []
        val_acc_list = []

        model.eval()

        for (val_data, val_target) in tqdm(val_loader):
            val_x, val_target = val_data.to(device), val_target.to(device)

            val_y_pred = model(val_x)

            val_loss, val_metric = criterion.loss_fn(val_target, val_y_pred)

            val_max_idx = torch.argmax(val_y_pred, dim=-1)
            val_loss_list.append(val_loss.detach().cpu().numpy())
            val_metric_list.append(val_metric.detach().cpu().numpy())
            val_acc = sum(val_target.detach().cpu().numpy()==val_max_idx.detach().cpu().numpy())/len(val_data)
            val_acc_list.append(val_acc)

    

    # TODO: Save in logfile
    avg_acc = sum(acc_list)/len(train_dataloader)
    avg_metric = sum(metric_list)/len(train_dataloader)
    avg_loss = sum(loss_list)/len(train_dataloader)
    avg_val_acc = sum(val_acc_list)/len(val_loader)
    avg_val_metric = sum(val_metric_list)/len(val_loader)
    avg_val_loss = sum(val_loss_list)/len(val_loader)
    print(f"[Epoch {epoch}] Fold {fold} Train Accuracy: {avg_acc}\t F1 Score: {avg_metric}\t Loss: {avg_loss}")
    print(f"[Epoch {epoch}] Fold {fold} Validation Accuracy: {avg_val_acc}\t F1 Score: {avg_val_metric}\t Loss: {avg_val_loss}")
    if min_loss > avg_loss and  min_val_loss > avg_val_loss:
        torch.save(model, f'./checkpoints/Epoch{epoch}_Fold{fold}model.pt')
    return min(min_loss, avg_loss), min(min_val_loss, avg_val_loss)

