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

from torchsummary import summary

## kfold
from sklearn.model_selection import StratifiedKFold
from kfoldtrain import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification', add_help=False)

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
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')

    # resume
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--checkpoint', default='', type=str)

    # seed
    parser.add_argument('--seed', default=42, type=int)

    # k-fold validation
    parser.add_argument('--n_splits', default=5, type=int)

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

# 한 fold 당 모델을 initialize함
def modelinit():
    #TODO argparser로 모델 바꾸는거 가져와야함.
    model = EfficientNet.from_pretrained('efficientnet-b7')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features=in_features, out_features=args.classes)
    return model


def main(args):
    if not os.path.isdir(os.path.join(os.getcwd(), 'checkpoints')):
        os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
    # image size: (384, 512)
    # image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=(0.558, 0.512, 0.478), std=(0.218, 0.238, 0.252)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # Loading traindataset
    train_dataset = TrainDataset(transform=transform, classes=args.classes, tr=args.target)

    criterion = Criterion()
    
    
    
    skf = StratifiedKFold(n_splits=args.n_splits)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset.img_paths, train_dataset.labels)):
        model = modelinit()
        # Optimizer
        if args.sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        model.to(device)
        min_loss = 100
        min_val_loss = 1000
        train_dataloader, val_dataloader = getKfoldDataloaders(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, train_idx=train_idx, val_idx=val_idx)
        print(f'FOLD {fold}')
        for epoch in range(args.epochs):
            print(f"Epoch {epoch} training")
            min_loss, min_val_loss = ktrain(model, train_dataloader, val_dataloader, optimizer, criterion, epoch, device, min_loss, min_val_loss, fold)


    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)