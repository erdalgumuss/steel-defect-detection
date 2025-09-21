import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Any # Add 'Any' to the import list

class SegmentationDataset(Dataset):
    """
    Generic dataset for image segmentation.
    - Reads images and class-wise masks
    - Applies augmentations if provided
    """

    def __init__(self, split_file: str, img_dir: str, mask_dir: str, augmentations: Optional[Any] = None, num_classes: int = 4):
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        # --- Load image ---
        img_path = os.path.join(self.img_dir, image_id)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, _ = image.shape

        # --- Load masks (channel-first) ---
        masks = []
        for c in range(1, self.num_classes + 1):
            mask_path = os.path.join(
                self.mask_dir, f"class{c}", image_id.replace(".jpg", f"_c{c}.png")
            )
            mask_c = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_c is None:
                mask_c = np.zeros((H, W), dtype=np.uint8)
            masks.append(mask_c)

        mask = np.stack(masks, axis=0).astype("float32") / 255.0

        # --- Apply augmentations ---
        if self.augmentations:
            augmented = self.augmentations(image=image, masks=[mask[c] for c in range(self.num_classes)])
            image = augmented["image"]
            mask = np.stack(augmented["masks"], axis=0)
            mask = torch.from_numpy(mask).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        if idx == 0:
            print(f"[DEBUG] image shape: {image.shape}, mask shape: {mask.shape}")
            assert mask.ndim == 3 and mask.shape[0] == self.num_classes, \
                f"Mask shape wrong: {mask.shape}"

        return image, mask

class SteelDefectDataset(SegmentationDataset):
    """Dataset wrapper for Steel Defect Detection."""
    pass