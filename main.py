import os
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from model import Model
from dataset import TrainDataset
from loss import Criterion
from train import train
from transformation import get_transform
from model import EfficientNet
from label_smoothing_loss import LabelSmoothing

from torch.utils.tensorboard import SummaryWriter

## time for tensorboard log
from datetime import datetime

from early_stopping import EarlyStopping


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification', add_help=False)

    # choose model version
    # larger the number larger the number of param choose from 0 to 7
    parser.add_argument('--model', default=0, type=int)
    
    # number of classes
    parser.add_argument('--classes', default=18, type=int)
    parser.add_argument('--target', default='mask', type=str)

    # transformation
    parser.add_argument('--tf', default='americano', type=str)

    # hyperparameters
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str)

    # resume
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--checkpoint', default='', type=str)

    # seed
    parser.add_argument('--seed', default=42, type=int)

    # early-stopping
    parser.add_argument('--es', default=True, type=bool)
    parser.add_argument('--patience', default=4, type=int)

    # label-smoothing
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--smoothing_level', default=0.1, type=float)

    # cut_mix
    parser.add_argument('--cut_mix', action='store_true')
    parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')

    return parser


# ????????? seed??? ??????
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

    if args.cut_mix and not os.path.isdir(os.path.join(os.getcwd(), 'checkpoints', 'cut_mix')):
        os.mkdir(os.path.join(os.getcwd(), 'checkpoints', 'cut_mix'))

    # image transformation
    transform = get_transform(args.tf)

    # Convert time zone
    now = datetime.now()
    n_time = now.strftime("%m_%d_%H:%M")

    writer = SummaryWriter(f'runs/{n_time}_b{args.model}_tf:{args.tf}_lr:{args.lr}_bs:{args.batch_size}_epochs_{args.epochs}')

    # Loading traindataset
    train_set = TrainDataset(transform=transform, classes=args.classes, tr=args.target, train=True)
    validation_set = TrainDataset(transform=transform, classes=args.classes, tr=args.target+'_infer', train=False)

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
    
    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    elif args.scheduler == 'multiply':
        lmbda = lambda epoch: 0.98739
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    if args.label_smoothing:
        criterion = LabelSmoothing(args.smoothing_level)
    else:
        criterion = Criterion()

    model.to(device)
    min_val_loss = 1e9
    global_step = 0

    if args.es:
        if args.cut_mix:
            early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=f'./checkpoints/cut_mix/{n_time}_model{args.model}_early_stopped_checkpoint.pt')
        else:
            early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=f'./checkpoints/{n_time}_model{args.model}_early_stopped_checkpoint.pt')
    else:
        early_stopping = None

    for epoch in range(args.epochs):
        # training
        print(f"Epoch {epoch} training")
        min_val_loss, avg_acc, avg_metric, avg_loss, val_avg_acc, val_avg_metric, val_avg_loss = train(model, train_dataloader, validation_dataloader, optimizer, criterion, epoch, device, min_val_loss, writer, global_step, lr_scheduler, early_stopping, args, n_time)

        writer.add_scalar("Training Accuracy", avg_acc, epoch)
        writer.add_scalar("Training F1 score", avg_metric, epoch)
        writer.add_scalar("Training Loss", avg_loss, epoch)
        writer.add_scalar("Validation Accuracy", val_avg_acc, epoch)
        writer.add_scalar("Validation F1 score", val_avg_metric, epoch)
        writer.add_scalar("Validation Loss", val_avg_loss, epoch)
        global_step += 1

        if early_stopping.early_stop:
            print(f"[Epoch {epoch}] Early stopped")
            break

    writer.flush()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)