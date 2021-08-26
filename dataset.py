import os
from glob import glob

from PIL import Image

from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self,  transform=None, cls=18, tr='mask'):
        self.cls = cls
        print(cls)
        self.tr = tr
        self.transform = transform
        self.img_paths = self.get_imgpath(cls, tr)
        self.labels = self.get_label(self.img_paths)
        self.classes = self.get_class(cls, tr)

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
        print()
        if cls == 3:
            if tr == 'mask':
                return ['correct', 'incorrect', 'no']
            elif tr == 'gender':
                return ['female','male']
            else:
                return ['under30','30to60','from60']
        else:
            return [i for i in range(18)]

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform: # transformation 적용
            image = self.transform(image)
        print('----', self.labels)
        print('---', self.classes.index(self.labels[index]))
        return image, self.classes.index(self.labels[index])

    def __len__(self):
        return len(self.img_paths)


