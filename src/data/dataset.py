# src/data/dataset.py
from __future__ import annotations
import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from .rle_decoder import build_multilabel_mask

from glob import glob
# src/data/dataset.py
class SteelDefectDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 image_dir: str,
                 shape: Tuple[int, int],          # 🔥 config’ten direkt alınıyor
                 num_classes: int = 4,
                 transforms=None,
                 cache_dir: str | None = None,
                 assume_fortran_order: bool = True,
                 load_rgb: bool = True,
                 force_resize: bool = False):     # 🔥 yeni param
        super().__init__()
        self.df = df.copy()
        self.image_dir = image_dir
        self.shape = shape
        self.num_classes = num_classes
        self.transforms = transforms
        self.cache_dir = cache_dir
        self.order = 'F' if assume_fortran_order else 'C'
        self.load_rgb = load_rgb
        self.force_resize = force_resize

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        if 'ClassId' not in self.df.columns:
            self.df['ClassId'] = np.nan

        self.groups: List[tuple[str, pd.DataFrame]] = list(self.df.groupby('ImageId'))

    def __len__(self) -> int:
        return len(self.groups)

    def _read_image(self, image_id: str) -> np.ndarray:
        path = os.path.join(self.image_dir, image_id)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if self.load_rgb:
            img = np.repeat(img[..., None], 3, axis=2)  # (H,W,3)
        return img

    def _read_mask(self, image_id: str, rows_for_image: pd.DataFrame) -> np.ndarray:
        mask = build_multilabel_mask(rows_for_image,
                                     shape=self.shape,
                                     num_classes=self.num_classes,
                                     order=self.order)
        return mask

    def __getitem__(self, idx: int):
        image_id, rows = self.groups[idx]
        img = self._read_image(image_id)
        mask = self._read_mask(image_id, rows)  # (C,H,W)

        mask_hwc = np.transpose(mask, (1,2,0))  # (H,W,C)

        # 🔥 Boyut check
        if img.shape[:2] != mask_hwc.shape[:2]:
            if self.force_resize:
                mask_hwc = cv2.resize(mask_hwc, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            else:
                raise ValueError(f"Image/mask size mismatch: "
                                 f"img={img.shape[:2]}, mask={mask_hwc.shape[:2]}, id={image_id}")

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask_hwc)
            img, mask_hwc = out['image'], out['mask']
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()
            mask_hwc = torch.from_numpy(mask_hwc).float()

        if isinstance(mask_hwc, np.ndarray):
            mask = torch.from_numpy(np.transpose(mask_hwc,(2,0,1))).float()
        else:
            mask = mask_hwc.permute(2,0,1).float()

        return img, mask, {"image_id": image_id}


# --------- Convenience function for building a merged df (optional) ---------

def build_full_dataframe(train_df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    Given raw train_df and image directory, returns a merged df containing all images
    present on disk with their (possibly NaN) EncodedPixels & ClassId rows.

    This mirrors the merge you've done in your notebook.
    """
    image_paths = glob(os.path.join(image_dir, "*.jpg"))
    image_ids = [os.path.basename(p) for p in image_paths]
    all_images_df = pd.DataFrame(image_ids, columns=["ImageId"])
    full_df = all_images_df.merge(train_df, on="ImageId", how="left")
    return full_df
