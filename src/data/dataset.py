# src/data/dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """
    Generic dataset for image segmentation.
    - Reads RGB images and multi-class binary masks
    - Applies Albumentations transforms if provided
    - Returns tensors in shape (C,H,W), float32
    """

    def __init__(
        self,
        split_file: str,
        img_dir: str,
        mask_dir: str,
        augmentations: Optional[Any] = None,
        num_classes: int = 4,
        img_ext: str = ".jpg",
        mask_ext: str = ".png",
    ):
        with open(split_file, "r") as f:
            self.image_ids: List[str] = [line.strip() for line in f.readlines()]

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.num_classes = num_classes
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: str) -> np.ndarray:
        """Load RGB image."""
        img_path = os.path.join(self.img_dir, image_id)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_masks(self, image_id: str, H: int, W: int) -> np.ndarray:
        """Load per-class binary masks and stack as (C,H,W)."""
        masks: List[np.ndarray] = []
        base_name = image_id.replace(self.img_ext, "")

        for c in range(1, self.num_classes + 1):
            mask_path = os.path.join(
                self.mask_dir, f"class{c}", f"{base_name}_c{c}{self.mask_ext}"
            )
            mask_c = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_c is None:
                mask_c = np.zeros((H, W), dtype=np.uint8)
            masks.append(mask_c)

        mask = np.stack(masks, axis=0).astype("float32") / 255.0
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]

        image = self._load_image(image_id)
        H, W, _ = image.shape
        mask = self._load_masks(image_id, H, W)

        if self.augmentations:
            augmented = self.augmentations(
                image=image, masks=[mask[c] for c in range(self.num_classes)]
            )
            image = augmented["image"]
            mask = torch.stack(
                [torch.as_tensor(m, dtype=torch.float32) for m in augmented["masks"]]
            )
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        if idx == 0:  # sadece ilk batch için debug
            logger.debug(f"Image shape: {tuple(image.shape)}, Mask shape: {tuple(mask.shape)}")
            assert mask.shape[0] == self.num_classes, \
                f"Expected {self.num_classes} mask channels, got {mask.shape[0]}"

        return image, mask


class SteelDefectDataset(SegmentationDataset):
    """Dataset wrapper for Steel Defect Detection."""
    pass
