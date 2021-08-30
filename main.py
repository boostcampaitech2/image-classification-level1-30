import os
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

# from model import Model
from dataset import TrainDataset
from loss import Criterion
from train import train

from model import EfficientNet
from loss import Criterion

from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

## validation set
from sklearn.model_selection import StratifiedShuffleSplit


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification', add_help=False)

    # choose model version
    # larger the number larger the number of param choose from 0 to 7
    parser.add_argument('--model', default=0, type=int)
    
    # number of classes
    parser.add_argument('--classes', default=18, type=int)
    parser.add_argument('--target', default='mask', type=str)

    # hyperparameters
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # resume
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--checkpoint', default='', type=str)

    # seed
    parser.add_argument('--seed', default=42, type=int)

    # checkpoints

    return parser

# 하나의 seed로 고정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True      

def main(args):
    if not os.path.isdir(os.path.join(os.getcwd(), 'checkpoints')):
        os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
    # image size: (384, 512)
    # image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),        
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.558, 0.512, 0.478), std=(0.218, 0.238, 0.252)),
    ])

    transform_albu = A.Compose(
    [  
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ])        
    
    writer = SummaryWriter()

    # stratified validation set maker
    validation_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    # Loading traindataset
    train_dataset = TrainDataset(transform=transform, classes=args.classes, tr=args.target)
    
    # split train dataset with stratified shuffle split, return indices
    for training_set_index, validation_set_index in validation_splitter.split(train_dataset.img_paths, train_dataset.labels):
        train_set = torch.utils.data.Subset(train_dataset, indices=training_set_index)
        validation_set = torch.utils.data.Subset(train_dataset, indices=validation_set_index)

    # make dataloader
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model
    model = EfficientNet.from_pretrained(f'efficientnet-b{args.model}')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features=in_features, out_features=args.classes)
    # print(model)

    # Optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)


    criterion = Criterion()

    model.to(device)
    min_val_loss = 100
    global_step = 0
    for epoch in range(args.epochs):
        # training
        print(f"Epoch {epoch} training")
        min_val_loss, avg_acc, avg_metric, avg_loss, val_avg_acc, val_avg_metric, val_avg_loss = train(model, train_dataloader, validation_dataloader, optimizer, criterion, epoch, device, min_val_loss, writer, global_step, lr_scheduler)

        writer.add_scalar("Training Accuracy", avg_acc, epoch)
        writer.add_scalar("Training F1 score", avg_metric, epoch)
        writer.add_scalar("Training Loss", avg_loss, epoch)
        writer.add_scalar("Validation Accuracy", val_avg_acc, epoch)
        writer.add_scalar("Validation F1 score", val_avg_metric, epoch)
        writer.add_scalar("Validation Loss", val_avg_loss, epoch)
        global_step += 1
    
    writer.flush()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)