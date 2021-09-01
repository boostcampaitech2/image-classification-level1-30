import os
import pandas as pd
from PIL import Image
import argparse
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from transformation import get_transform


def get_args_parser():
    parser = argparse.ArgumentParser('transformation', add_help=False)

    # choose transform version
    parser.add_argument('--tf', default='americano', type=str)

    return parser

parser = argparse.ArgumentParser(description='transformation', parents=[get_args_parser()])
args = parser.parse_args()


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(self.img_paths[index].strip()), cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)


# 모델 불러오기
test_dir = '/opt/ml/input/data/eval'
device = torch.device('cuda')
model = torch.load('./checkpoints/09_01_14:50_Epoch3_val_F10.773_val_acc87.25%model.pt').to(device)
print(model)
model.eval()


# meta 데이터와 이미지 경로를 불러오기
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체, DataLoader를 생성
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

transform = get_transform(args.tf+'_infer')

dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

# 결과 예측
all_predictions = []
for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일 저장
submission.to_csv(os.path.join(test_dir, 'submission_NoTTA.csv'), index=False)
print('test inference is done!')