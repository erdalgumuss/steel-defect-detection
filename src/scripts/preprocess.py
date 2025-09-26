#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.rle_decoder import rle_to_mask

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def overlay_mask_on_image(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4):
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def main(args):
    H, W = args.image_height, args.image_width

    train_csv = os.path.join(args.data_dir, "train.csv")
    image_dir = os.path.join(args.data_dir, "train_images")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    df = pd.read_csv(train_csv)

    if "ImageId_ClassId" in df.columns:
        tmp = df["ImageId_ClassId"].str.split("_", expand=True)
        df["ImageId"] = tmp[0]
        df["ClassId"] = tmp[1].astype(int)

    df["EncodedPixels"] = df["EncodedPixels"].fillna("")
    df["ClassId"] = df["ClassId"].astype(int)

    out_masks = os.path.join(args.out_dir, "masks_npz")
    out_overlays = os.path.join(args.out_dir, "overlays")
    out_reports = os.path.join(args.out_dir, "reports")
    out_splits = os.path.join(args.out_dir, "splits")

    for d in [args.out_dir, out_masks, out_overlays, out_reports, out_splits]:
        ensure_dir(d)

    image_ids = df["ImageId"].unique().tolist()
    cls_counts = defaultdict(int)

    print(f"[INFO] Processing {len(image_ids)} images...")

    for image_id in tqdm(image_ids):
        masks = []
        for c in [1, 2, 3, 4]:
            rle = df.loc[(df.ImageId == image_id) & (df.ClassId == c), "EncodedPixels"]
            rle = rle.values[0] if len(rle) else ""
            mask_c = rle_to_mask(rle, shape=(H, W))
            masks.append(mask_c)
            if mask_c.sum() > 0:
                cls_counts[c] += 1

        masks = np.stack(masks, axis=-1)  # (H, W, 4)
        base = os.path.splitext(image_id)[0]
        np.savez_compressed(os.path.join(out_masks, f"{base}.npz"), mask=masks.astype(np.uint8))

        if args.num_visualize > 0 and np.random.rand() < (args.num_visualize / len(image_ids)):
            img_path = os.path.join(image_dir, image_id)
            img = cv2.imread(img_path)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            out = img.copy()
            for i in range(4):
                if masks[..., i].any():
                    out = overlay_mask_on_image(out, masks[..., i], color=colors[i])
            cv2.imwrite(os.path.join(out_overlays, f"{base}_overlay.jpg"), out)

    # Split üret (stratify any_defect)
    print("[INFO] Creating stratified train/val split...")
    img_df = df.groupby("ImageId")["EncodedPixels"].apply(
        lambda x: any([len(s) > 0 for s in x])
    ).astype(int).reset_index(name="any_defect")

    train_ids, val_ids = train_test_split(
        img_df["ImageId"], 
        test_size=args.val_split, 
        stratify=img_df["any_defect"], 
        random_state=42
    )

    with open(os.path.join(out_splits, "train.txt"), "w") as f:
        f.write("\n".join(train_ids.tolist()))
    with open(os.path.join(out_splits, "val.txt"), "w") as f:
        f.write("\n".join(val_ids.tolist()))

    # rapor
    report = {
        "num_images": len(image_ids),
        "class_positive_counts": dict(cls_counts),
        "train_size": len(train_ids),
        "val_size": len(val_ids)
    }
    with open(os.path.join(out_reports, "preprocess_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("[INFO] Done. Report written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=1600)
    parser.add_argument("--num_visualize", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    main(args)
