import os
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.csv_list = ['ages.csv', 'genders.csv', 'masks.csv']
        self.img_paths = list(pd.read_csv(os.path.join(path, 'images.csv')))
        self.labels = list(map(float, pd.read_csv(os.path.join(path, 'labels.csv')))) # ages.csv, genders.csv, masks.csv
        # print(self.labels)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform: # transformation 적용
            image = self.transform(image)

        label = int(self.labels[index])
        return image, label

    def __len__(self):
        return len(self.img_paths)