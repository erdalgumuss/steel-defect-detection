# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def build_transforms(phase: str, img_size: Tuple[int, int] = (256, 1600)) -> A.Compose:
    """Return Albumentations Compose for train/val."""
    common = [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True),
    ]

    if phase == "train":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ]
        return A.Compose(aug + common)
    elif phase == "val":
        return A.Compose(common)
    else:
        raise ValueError(f"Unknown phase: {phase}")
