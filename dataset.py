import os
from glob import glob

from PIL import Image

from torch.utils.data import Dataset

import albumentations as A
import cv2


IDX_NONE_TF = 0  # Index of No Transfrom (Initial Value)
IDX_TV_TF   = 1  # Index of TorchVision Transform
IDX_ALBU_TF = 2  # Index of Albumentation Transform

class TrainDataset(Dataset):
    def __init__(self,  transform=None, classes=18, tr='mask', train=True):
        super().__init__()
        self.cls = classes
        self.tr = tr
        self.train = train
        self.transform = transform
        self.img_paths = self.get_imgpath(classes, tr, train)
        self.labels = self.get_label(self.img_paths)
        self.classes = self.get_class(classes, tr)
        self.index_transform = IDX_ALBU_TF if isinstance(self.transform, A.Compose) else IDX_TV_TF
        
    def get_label(self, paths):
        label_list = []
        for p in paths:
            label_list.append(p.split('/')[-2])
        return label_list

    def get_imgpath(self, cls, tr, train):
        if train == True:
            if cls == 18:
                return glob(os.path.join(f'/opt/ml/input/data/train_{cls}class', '*/**'))
            else:
                return glob(os.path.join(f'/opt/ml/input/data/train_{cls}class/{tr}', '*/**'))
        else:
            if cls == 18:
                return glob(os.path.join(f'/opt/ml/input/data/val_{cls}class', '*/**'))
            else:
                return glob(os.path.join(f'/opt/ml/input/data/val_{cls}class/{tr}', '*/**'))

    def get_class(self, cls, tr):
        if cls == 3 or cls == 2:
            if tr == 'mask':
                return ['correct', 'incorrect', 'no']
            elif tr == 'gender':
                return ['female','male']
            else:
                return ['under30','30to60','from60']
        else:
            return [str(i) for i in range(18)]

    def __getitem__(self, index):
        if self.index_transform == IDX_ALBU_TF:
            image = cv2.cvtColor(cv2.imread(self.img_paths[index].strip()), cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)['image'] #dtype: uint8
        else:
            image = self.transform(Image.open(self.img_paths[index].strip()))
            
        return image, self.classes.index(self.labels[index])

    def __len__(self):
        return len(self.img_paths)
