import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SteelDefectDataset(Dataset):
    """
    Çelik üretimindeki yüzey kusurları için dataset sınıfı.
    Görselleri ve maskeleri okur, augmentations uygular.
    """

    def __init__(self, split_file, img_dir, mask_dir, augmentations=None):
        """
        Args:
            split_file (str): train.txt veya val.txt dosyası
            img_dir (str): Görsellerin bulunduğu dizin
            mask_dir (str): Kanal bazlı maskelerin bulunduğu dizin
            augmentations (albumentations.Compose): Veri artırma pipeline
        """
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Görseli oku
        img_path = os.path.join(self.img_dir, image_id)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Kanal bazlı maskeler
        masks = []
        for c in range(1, 5):  # class1..class4
            mask_path = os.path.join(
                self.mask_dir, f"class{c}", image_id.replace(".jpg", f"_c{c}.png")
            )
            mask_c = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_c is None:
                H, W, _ = image.shape
                mask_c = np.zeros((H, W), dtype=np.uint8)
            masks.append(mask_c)


        mask = np.stack(masks, axis=0).astype("float32") / 255.0  # (4, H, W)

        # Augmentation uygula
        if self.augmentations:
            augmented = self.augmentations(image=image, masks=[m for m in mask])
            image = augmented["image"]
            mask = np.stack(augmented["masks"], axis=0)

        return image, mask
