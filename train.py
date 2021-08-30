import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import logging


def train(model, train_dataloader, validation_dataloader, optimizer, criterion, epoch, device, min_val_loss, writer, global_step, lr_scheduler, early_stopping):
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

        # For each epoch, if batch_idx is multiple of 5, write five image dataset in tfboard
        if batch_idx % 10 == 0:
            writer.add_images(f'step:{global_step}\ty_pred:[{max_idx[0:5]}\ty_atcual:[{target[0:5]}]]', data[0:5,:,:,:], global_step=global_step)

        loss_list.append(train_loss.detach().cpu().numpy())
        metric_list.append(train_metric)
        acc = sum(target.detach().cpu().numpy()==max_idx.detach().cpu().numpy())/len(data)
        acc_list.append(acc)

        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        lr_scheduler.step()
    

    # Validation
    model.eval()

    val_loss_list = []
    val_metric_list = []
    val_acc_list = []
    
    for val_batch_idx, (val_data, val_target) in enumerate(tqdm(validation_dataloader)):
        val_x, val_target = val_data.to(device), val_target.to(device)

        # Forward
        val_y_pred = model(val_x)
        
        # Computing Loss
        val_train_loss, val_train_metric = criterion.loss_fn(val_target, val_y_pred)

        val_max_idx = torch.argmax(val_y_pred, dim=-1)

        # For each epoch, if batch_idx is multiple of 5, write five image dataset in tfboard
        if val_batch_idx % 10 == 0:
            writer.add_images(f'step:{global_step}\tval_y_pred:[{val_max_idx[0:5]}\tval_y_atcual:[{val_target[0:5]}]]', val_data[0:5,:,:,:], global_step=global_step)

        val_loss_list.append(val_train_loss.detach().cpu().numpy())
        val_metric_list.append(val_train_metric)
        val_acc = sum(val_target.detach().cpu().numpy()==val_max_idx.detach().cpu().numpy())/len(val_data)
        val_acc_list.append(val_acc)



    avg_acc = sum(acc_list)/len(train_dataloader)
    avg_metric = sum(metric_list)/len(train_dataloader)
    avg_loss = sum(loss_list)/len(train_dataloader)
    val_avg_acc = sum(val_acc_list)/len(validation_dataloader)
    val_avg_metric = sum(val_metric_list)/len(validation_dataloader)
    val_avg_loss = sum(val_loss_list)/len(validation_dataloader)
    print(f"[Epoch {epoch}] Train Accuracy: {avg_acc}\t F1 Score: {avg_metric}\t Loss: {avg_loss}\nValidation Accuracy: {val_avg_acc}\t Validation F1 Score: {val_avg_metric}\t Validation Loss: {val_avg_loss}")

    logging.basicConfig(filename='Performance.log',level=logging.INFO)
    logging.info(f"[Epoch {epoch}] Train Accuracy: {avg_acc}\t F1 Score: {avg_metric}\t Loss: {avg_loss}\nValidation Accuracy: {val_avg_acc}\t Validation F1 Score: {val_avg_metric}\t Validation Loss: {val_avg_loss}\n")


    if min_val_loss > val_avg_loss:
        torch.save(model, f'./checkpoints/Epoch{epoch}_val_F1{val_avg_metric:.3f}_val_acc{val_avg_acc:4.2%}model.pt')

    if early_stopping is not None:
        early_stopping(val_avg_loss, model)

    return min(min_val_loss, val_avg_loss), avg_acc, avg_metric, avg_loss, val_avg_acc, val_avg_metric, val_avg_loss