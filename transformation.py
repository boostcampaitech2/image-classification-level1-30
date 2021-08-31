import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

def get_transform(t_name):
    
    if t_name == 'water':
        t = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'sparkling_water':
        t = A.Compose(
        [   
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ])
        return t

    elif t_name =='americano':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'latte':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return t

    # inference

    elif t_name == 'water_infer':
        t = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=0, std=1),
            A.OneOf([
                A.MotionBlur(p=1),
                A.OpticalDistortion(p=1),
                A.GaussNoise(p=1)
            ], p=1),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'americano_infer' or t_name=='sparkling_water_infer':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=0, std=1),
            A.OneOf([
                A.MotionBlur(p=1),
                A.OpticalDistortion(p=1),
                A.GaussNoise(p=1)
            ], p=1),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'latte_infer':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.OneOf([
                A.MotionBlur(p=1),
                A.OpticalDistortion(p=1),
                A.GaussNoise(p=1)
            ], p=1),
            ToTensorV2(),
        ])
        return t
    
