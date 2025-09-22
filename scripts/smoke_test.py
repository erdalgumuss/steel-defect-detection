import os
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SteelDefectDataset
from src.data.transforms import build_transforms


def main():
    # --- Config ---
    split_file = "data/processed/splits/train_mini.txt"
    img_dir = "data/raw/train_images"
    mask_dir = "data/processed/masks_png"
    img_size = (256, 1600)
    batch_size = 2

    # --- Dataset ---
    dataset = SteelDefectDataset(
        split_file=split_file,
        img_dir=img_dir,
        mask_dir=mask_dir,
        augmentations=build_transforms("train", img_size),
        num_classes=4,
    )

    # --- DataLoader ---
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Iterate 1 batch ---
    images, masks = next(iter(loader))

    print("✅ Smoke test başarılı!")
    print(f"Images shape: {tuple(images.shape)} | dtype: {images.dtype}")
    print(f"Masks  shape: {tuple(masks.shape)} | dtype: {masks.dtype}")

    # --- Assertions ---
    assert images.ndim == 4, "Images must be 4D (B,C,H,W)"
    assert masks.ndim == 4, "Masks must be 4D (B,C,H,W)"
    assert images.shape[1] == 3, "Images must have 3 channels (RGB)"
    assert masks.shape[1] == 4, "Masks must have 4 channels (num_classes)"
    assert images.shape[2:] == img_size, f"Images resized to {img_size}"
    assert masks.shape[2:] == img_size, f"Masks resized to {img_size}"

    print("🎉 All assertions passed!")


if __name__ == "__main__":
    main()
