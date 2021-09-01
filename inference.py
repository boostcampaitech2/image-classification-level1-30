import os
import pandas as pd
from PIL import Image
import argparse
import cv2
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from transformation import get_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_args_parser():
    parser = argparse.ArgumentParser('transformation', add_help=False)

    # choose transform version
    parser.add_argument('--tf', default='americano', type=str)
    parser.add_argument('--tta', default=False, type=bool)

    return parser

parser = argparse.ArgumentParser(description='transformation', parents=[get_args_parser()])
args = parser.parse_args()
print(args)

class TestDataset(Dataset):
    def __init__(self, img_paths, transform, tta):
        self.img_paths = img_paths
        self.transform = transform
        self.tta = tta
        self.trans1 = get_transform('TTABLUR')
        self.trans2 = get_transform('TTAContrast')
        self.trans3 = get_transform('TTAFlipped')
        self.trans4 = get_transform('TTABright')

    def __getitem__(self, index):
        if self.tta:
            image = cv2.cvtColor(cv2.imread(self.img_paths[int(index/5)].strip()), cv2.COLOR_BGR2RGB)

            tmp = index % 5
            if tmp == 0:
                image = self.transform(image=image)['image']
            elif tmp == 1:
                image = self.trans1(image=image)['image']
            elif tmp == 2:
                image = self.trans2(image=image)['image']
            elif tmp == 3:
                image = self.trans3(image=image)['image']
            else:
                image = self.trans4(image=image)['image']
        else:
            image = cv2.cvtColor(cv2.imread(self.img_paths[index].strip()), cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image=image)['image']

        return image

    def __len__(self):
        if self.tta:
            return len(self.img_paths) * 5
        else:
            return len(self.img_paths)


# 모델 불러오기
test_dir = '/opt/ml/input/data/eval'
device = torch.device('cuda')
model = torch.load('./checkpoints/09_01_06:03_Epoch4_val_F10.680_val_acc79.27%model.pt').to(device)
# print(model)
model.eval()


# meta 데이터와 이미지 경로를 불러오기
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체, DataLoader를 생성
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

transform = get_transform(args.tf+'_infer')

print(args.tta)
dataset = TestDataset(image_paths, transform, args.tta)

if args.tta:
    loader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=4
    )

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = torch.mean(pred, axis=-2)
            pred = pred.argmax(dim=-1)
            all_predictions.append(pred.cpu().numpy())
    submission['ans'] = all_predictions

    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference with TTA is done!')
else:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    submission.to_csv(os.path.join(test_dir, 'submission_NoTTA.csv'), index=False)
    print('test inference No TTA is done!')