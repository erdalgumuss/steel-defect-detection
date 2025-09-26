# src/data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple


class SteelDefectDataset(Dataset):
    """
    PyTorch Dataset for steel defect segmentation with preprocessed .npz masks.
    Preprocess aşamasında üretilmiş split dosyaları (train.txt, val.txt) ve
    .npz mask klasörü ile çalışır.

    Args:
        split_ids: list of image_ids (örn: ['0001.jpg', '0002.jpg'])
        image_dir: ham görüntülerin klasörü
        mask_dir: preprocess.py tarafından üretilen .npz mask klasörü
        transforms: Albumentations pipeline
    """

    def __init__(
        self,
        split_ids: List[str],
        image_dir: str,
        mask_dir: str,
        transforms=None,
        load_rgb: bool = True,
    ):
        self.image_ids = split_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.load_rgb = load_rgb

    def __len__(self) -> int:
        return len(self.image_ids)

    def _read_image(self, image_id: str) -> np.ndarray:
        path = os.path.join(self.image_dir, image_id)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if self.load_rgb:
            img = np.repeat(img[..., None], 3, axis=2)  # (H, W, 3)
        return img

    def _read_mask(self, image_id: str) -> np.ndarray:
        base = os.path.splitext(image_id)[0]
        path = os.path.join(self.mask_dir, f"{base}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask not found: {path}")
        mask = np.load(path)["mask"]  # (H, W, C)
        return mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img = self._read_image(image_id)
        mask = self._read_mask(image_id)  # (H, W, C)

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            # Albumentations ToTensorV2 -> mask: (H, W, C)
            if mask.ndim == 3 and mask.shape[-1] == 4:
                mask = mask.permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        meta = {"image_id": image_id}
        return img, mask, meta
