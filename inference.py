import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


# 모델 불러오기
test_dir = '/opt/ml/input/data/eval'
device = torch.device('cuda')
model = torch.load('./checkpoints/model.pt').to(device)
print(model)
model.eval()


# meta 데이터와 이미지 경로를 불러오기
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체, DataLoader를 생성
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((224, 224), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.558, 0.512, 0.478), std=(0.218, 0.238, 0.252)),
])

dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# 결과 예측
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일 저장
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')