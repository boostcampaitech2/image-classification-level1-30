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

root = '/opt/ml/'
# train_path = '/opt/ml/input/data/train's

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification', add_help=False)

    # hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argumet('--sgd', default=False, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)

    # seed
    parser.add_argument('--seed', default=42, type=int)

    # checkpoints
    parser.add_argument()

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
    # image size: (384, 512)

    # image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, 224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # Loading traindataset
    train_dataset = TrainDataset(root, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model
    model = EfficientNet.from_pretrained('efficientnet-b0')

    # Optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    criterion = loss.Criterion()

    model.to(device)
    
    ## Model test code
    # inputs = torch.rand(1, 3, 224, 224).to(device)
    # model = EfficientNet.from_pretrained('efficientnet-b0')
    # model.eval()
    # outputs = model(inputs)
    # print(outputs.shape)

    for epoch in range(args.epochs):
        # training
        # TODO: Logfile or Tensorboard 작성
        print(f"Epoch {epoch} training")
        train(model, train_dataloader, optimizer, criterion, epoch, device)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)