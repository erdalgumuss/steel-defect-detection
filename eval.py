import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import Config
from data.dataset import SteelDefectDataset, build_full_dataframe
from data.transforms import get_valid_transforms
from models.unet import UNetResNet18
from metrics.dice_coefficient import dice_coefficient, dice_per_class


@torch.no_grad()
def evaluate(config_path: str = "config.yaml", checkpoint: str = None, num_samples: int = 5):
    # ---- Config ----
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    # ---- Dataset ----
    train_csv = pd.read_csv(cfg["data"]["train_csv"])
    image_dir = cfg["data"]["train_images_dir"]
    full_df = build_full_dataframe(train_csv, image_dir)

    # 👇 Train/Valid Split (aynı train.py ile uyumlu)
    train_ids, valid_ids = train_test_split(
        full_df["ImageId"].unique(),
        test_size=cfg["data"]["val_split"],
        random_state=42,
        shuffle=True
    )
    valid_df = full_df[full_df["ImageId"].isin(valid_ids)].reset_index(drop=True)

    valid_ds = SteelDefectDataset(
        valid_df,
        image_dir,
        shape=(cfg["data"]["image_height"], cfg["data"]["image_width"]),
        num_classes=cfg["data"]["num_classes"],
        transforms=get_valid_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    )
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)

    # ---- Model ----
    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        decoder_mode=cfg["model"].get("decoder_mode", "add")
    ).to(device)

    if checkpoint is None:
        checkpoint = os.path.join(cfg["logging"]["output_dir"], "model_final.pth")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # ---- Eval loop ----
    dices, per_class = [], []
    samples = []

    for i, (images, masks, metas) in enumerate(valid_loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        # Metrics
        dice = dice_coefficient(outputs, masks).item()
        class_scores = dice_per_class(outputs, masks)
        dices.append(dice)
        per_class.append(class_scores)

        # Save a few samples
        if i < num_samples:
            samples.append((images[0].cpu(), masks[0].cpu(), outputs[0].cpu(), metas))

    # ---- Report ----
    mean_dice = sum(dices) / len(dices)
    per_class_mean = torch.tensor(per_class).mean(0).tolist()

    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Per-class Dice: {per_class_mean}")

    # ---- Save results to JSON ----
    results = {
        "mean_dice": mean_dice,
        "per_class_dice": per_class_mean,
        "checkpoint": checkpoint
    }
    results_path = Path(cfg["logging"]["output_dir"]) / "eval_results.json"
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved: {results_path}")

    # ---- Visualization ----
    for img, mask, pred, meta in samples:
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()
        pred_np = (torch.sigmoid(pred) > 0.5).permute(1, 2, 0).numpy()

        n_classes = mask_np.shape[-1]
        fig, axs = plt.subplots(3, n_classes, figsize=(5 * n_classes, 10))

        # Normalize image for display
        axs[0, 0].imshow((img_np - img_np.min()) / (img_np.max() - img_np.min()))
        axs[0, 0].set_title(f"Image {meta[0]['image_id'] if isinstance(meta, list) else meta['image_id']}")
        axs[0, 0].axis("off")
        for j in range(1, n_classes):
            axs[0, j].axis("off")

        for c in range(n_classes):
            axs[1, c].imshow(mask_np[..., c], cmap="gray")
            axs[1, c].set_title(f"GT Class {c+1}")
            axs[1, c].axis("off")

            axs[2, c].imshow(pred_np[..., c], cmap="gray")
            axs[2, c].set_title(f"Pred Class {c+1}")
            axs[2, c].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate()

 #### python src/eval.py --config config.yaml --checkpoint outputs/model_final.pth
