#src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def get_train_transforms(img_size: Tuple[int, int] = (256, 1600)) -> A.Compose:
    return A.Compose([
        A.Resize(*img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(img_size: Tuple[int, int] = (256, 1600)) -> A.Compose:
    return A.Compose([
        A.Resize(*img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])