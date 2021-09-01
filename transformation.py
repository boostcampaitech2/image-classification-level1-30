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

    elif t_name == 'yogurt':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'blueberry_yogurt':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.GaussianBlur(p=0.2),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8),
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
            ToTensorV2(),
        ])
        return t

    elif t_name == 'americano_infer' or t_name=='sparkling_water_infer':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'latte_infer' or t_name == 'blueberry_yogurt_infer' or t_name == 'yogurt_infer':
        t = A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return t
    
    elif t_name == 'TTABLUR':
        t =  A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.GaussianBlur(),
            ToTensorV2(),
        ])
        return t
    
    elif t_name == 'TTAContrast':
        t =  A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.RandomBrightnessContrast(),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'TTAFlipped':
        t =  A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.HorizontalFlip(),
            ToTensorV2(),
        ])
        return t

    elif t_name == 'TTABright':
        t =  A.Compose(
        [
            A.CenterCrop(height=400, width=384),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.GaussianBlur(),
            A.HorizontalFlip(),
            ToTensorV2(),
        ])
        return t