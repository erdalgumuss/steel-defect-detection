
# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(target_h=256, target_w=1600, crop_prob=0.5):
    return A.Compose([
        A.CropNonEmptyMaskIfExists(height=target_h, width=target_w, p=crop_prob),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.02, 0.02), rotate=(-5, 5), fit_output=False, p=0.5),
        # A.GaussNoise(...)  # sürüm uyuşmazlığı varsa şimdilik kaldır
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.Resize(target_h, target_w, interpolation=1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_valid_transforms(target_h: int = 256, target_w: int = 1600, *,
                         normalize_imagenet: bool = True) -> A.Compose:
    tfms = [
        A.Resize(target_h, target_w, interpolation=1),
    ]
    if normalize_imagenet:
        tfms += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    tfms += [ToTensorV2()]
    return A.Compose(tfms)


def get_infer_transforms(target_h: int = 256, target_w: int = 1600, *,
                         normalize_imagenet: bool = True) -> A.Compose:
    # identical to valid; kept separate for clarity/future changes
    return get_valid_transforms(target_h=target_h, target_w=target_w, normalize_imagenet=normalize_imagenet)


