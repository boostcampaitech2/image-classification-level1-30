import os
from glob import glob
import numpy as np

from PIL import Image

from torch.utils.data import Dataset

import albumentations as A    #pip install -U albumentations
from albumentations.pytorch import ToTensorV2
import cv2

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from matplotlib import pyplot as plt

IDX_NONE_TF = 0  #Index of No Transfrom (Initial Value)
IDX_TV_TF   = 1  #Index of TorchVision Transform
IDX_ALBU_TF = 2  #Index of Albumentation Transform

class TrainDataset(Dataset):
    def __init__(self,  transform=None, classes=18, tr='mask'):
        super().__init__()
        self.cls = classes
        self.tr = tr
        self.transform = transform
        self.img_paths = self.get_imgpath(classes, tr)
        self.labels = self.get_label(self.img_paths)
        self.classes = self.get_class(classes, tr)
        if self.transform:    #transform이 있을 경우 Albumentation or TorchVision Transformation인지를 저장
            self.index_transform = IDX_ALBU_TF if isinstance(self.transform, A.Compose) else IDX_TV_TF
        
    def get_label(self, paths):
        label_list = []
        for p in paths:
            label_list.append(p.split('/')[-2])
        return label_list

    def get_imgpath(self, cls, tr):
        if cls == 18:
            return glob(os.path.join(f'/opt/ml/input/data/train_{cls}class', '*/**'))
        else:
            return glob(os.path.join(f'/opt/ml/input/data/train_{cls}class/{tr}', '*/**'))

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
        if self.transform: #transform이 있을 경우 index에 따라 Albumantion Transformation or TorchVision transformation 적용
            if self.index_transform == IDX_ALBU_TF:
                image = cv2.cvtColor(cv2.imread(self.img_paths[index].strip()), cv2.COLOR_BGR2RGB)
                image = self.transform(image=image)['image'] #dtype: uint8
                image = transforms.ToTensor()(image) #dtype FloatTensor
            else:
                image = self.transform(Image.open(self.img_paths[index].strip()))
        else: #transform이 없을 경우   
            image = Image.open(self.img_paths[index].strip())
            
        return image, self.classes.index(self.labels[index])

    def __len__(self):
        return len(self.img_paths)

    def test_transform(self):
        path_test_image = '/opt/ml/input/data/train/images/000001_female_Asian_45/mask1.jpg'
        image_initial = Image.open(path_test_image)    
        if self.transform: #transform이 있을 경우 index_transform에 따라 Albumantion Transformation or TorchVision transformation 적용
            if self.index_transform == IDX_ALBU_TF:
                image = cv2.cvtColor(cv2.imread(path_test_image), cv2.COLOR_BGR2RGB)
                image = self.transform(image=image)['image']
                image = transforms.ToTensor()(image)
            else:
                image = self.transform(Image.open(path_test_image))
        fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        ax[0].set_title('original ' + str(image_initial.size) )
        ax[0].imshow(transforms.ToPILImage()(transforms.ToTensor()(image_initial)))     
        if self.transform:
            title_name = 'Albu ' if self.index_transform == IDX_ALBU_TF else 'T.V. '
            ax[1].set_title(title_name + str(image.shape))
            ax[1].imshow(transforms.ToPILImage()(image))                        
    
