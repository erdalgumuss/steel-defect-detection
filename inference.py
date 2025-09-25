import os
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from data.transforms import get_infer_transforms
from models.unet import UNetResNet18


# -------------------------
# RLE Encode
# -------------------------
def mask_to_rle(mask: np.ndarray, order: str = "F") -> str:
    """
    Encode binary mask to RLE.
    mask: (H, W) binary np.uint8
    """
    pixels = mask.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# -------------------------
# Single Image Inference
# -------------------------
@torch.no_grad()
def infer_single_image(model, image_path, device, cfg):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = np.repeat(img[..., None], 3, axis=2)  # grayscale → RGB

    tfms = get_infer_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    sample = tfms(image=img)
    tensor = sample["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).float().cpu().numpy()[0]  # (C,H,W)

    return mask


# -------------------------
# Main
# -------------------------
def main(config_path="config.yaml",
         checkpoint="outputs/model_final.pth",
         infer_dir="data/test_images",
         save_png=True,
         save_csv=True):
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    # Model
    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        decoder_mode=cfg["model"].get("decoder_mode", "add")
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # Output dirs
    out_dir = Path(cfg["logging"]["output_dir"]) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for img_name in os.listdir(infer_dir):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(infer_dir, img_name)
        mask = infer_single_image(model, img_path, device, cfg)

        # Save masks as PNG
        if save_png:
            for c in range(cfg["data"]["num_classes"]):
                out_path = out_dir / f"{img_name}_class{c+1}.png"
                cv2.imwrite(str(out_path), (mask[c] * 255).astype(np.uint8))

        # Save as RLE for Kaggle submission
        if save_csv:
            for c in range(cfg["data"]["num_classes"]):
                rle = mask_to_rle((mask[c] > 0.5).astype(np.uint8))
                if rle:  # only non-empty masks
                    results.append({
                        "ImageId": img_name,
                        "ClassId": c + 1,
                        "EncodedPixels": rle
                    })

    if save_csv:
        df = pd.DataFrame(results, columns=["ImageId", "ClassId", "EncodedPixels"])
        csv_path = out_dir / "submission.csv"
        df.to_csv(csv_path, index=False)
        print(f"Submission CSV saved: {csv_path}")

    print(f"Inference complete. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
