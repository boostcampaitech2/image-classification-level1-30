import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(model, train_dataloader, validation_dataloader, optimizer, criterion, epoch, device, min_val_loss, writer, global_step, lr_scheduler, early_stopping, args, n_time):
    # For recording time

    # For computing average of loss, metric, accuracy
    loss_list = []
    metric_list = []
    acc_list = []

    # Training
    model.train()
    
    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
        x, target = data.to(device), target.to(device)

        if args.cut_mix:
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # Generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(x.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

                # Adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                # Forward
                y_pred = model(x)

                # Compute loss, metric
                train_loss_a, train_metric_a = criterion.loss_fn(target_a, y_pred)
                train_loss_b, train_metric_b = criterion.loss_fn(target_b, y_pred)
                train_loss = train_loss_a * lam + train_loss_b * (1. - lam)
                train_metric = train_metric_a * lam + train_metric_b * (1. - lam)
                max_idx = torch.argmax(y_pred, dim=-1)
                acc = (sum(target_a.detach().cpu().numpy() == max_idx.detach().cpu().numpy())/len(data)) * lam + (sum(target_b.detach().cpu().numpy() == max_idx.detach().cpu().numpy())/len(data)) * (1. - lam)
            else:
                # Forward
                y_pred = model(x)
                
                # Compute loss, metric
                train_loss, train_metric = criterion.loss_fn(target, y_pred)
                max_idx = torch.argmax(y_pred, dim=-1)
                acc = sum(target.detach().cpu().numpy() == max_idx.detach().cpu().numpy()) / len(data)
            
            loss_list.append(train_loss.detach().cpu().numpy())
            metric_list.append(train_metric)
            acc_list.append(acc)
        else:
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
        if args.cut_mix:
            torch.save(model, f'./checkpoints/cut_mix/{n_time}_Epoch{epoch}_val_F1{val_avg_metric:.3f}_val_acc{val_avg_acc:4.2%}model.pt')
        else:
            torch.save(model, f'./checkpoints/{n_time}_Epoch{epoch}_val_F1{val_avg_metric:.3f}_val_acc{val_avg_acc:4.2%}model.pt')

    if early_stopping is not None:
        early_stopping(val_avg_loss, model)

    return min(min_val_loss, val_avg_loss), avg_acc, avg_metric, avg_loss, val_avg_acc, val_avg_metric, val_avg_loss