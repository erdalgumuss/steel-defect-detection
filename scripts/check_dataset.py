# scripts/check_dataset.py
import matplotlib.pyplot as plt
from src.data.dataset import SteelDefectDataset
from src.data.transforms import get_train_transforms
import os
def main():
    # Dataset yükle
    dataset = SteelDefectDataset(
        split_file="data/processed/splits/train.txt",   # preprocess sonrası oluşan train.txt
        img_dir="data/raw/train_images",                # orijinal görseller
        mask_dir="data/processed/masks_png",            # kanal bazlı maskeler (class1..class4)
        augmentations=get_train_transforms()
    )

    print(f"Dataset uzunluğu: {len(dataset)}")
    out_dir = "outputs/debug"
    os.makedirs(out_dir, exist_ok=True)

    # Birkaç örneği test et
    for idx in range(10):
        img, mask = dataset[idx]

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 5, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Image")

        for i in range(mask.shape[0]):
            plt.subplot(1, 5, i+2)
            plt.imshow(mask[i], cmap="gray")
            plt.title(f"Class {i+1}")

        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{idx}.png")   # ekledik
        plt.close()



if __name__ == "__main__":
    main()
