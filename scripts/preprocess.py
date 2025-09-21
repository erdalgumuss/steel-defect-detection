import os
import cv2
import json
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.data.preprocess_utils import (
    ensure_dir,
    rle_decode,
    overlay_mask_on_image,
    save_masks_png,
    make_train_val_split,
    generate_report,
    is_empty_rle,
)


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config bulunamadı: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config


def main():
    H, W = 256, 1600  # dataset sabit boyutu
    config = load_config()

    # ------------------------
    # Config parametreleri
    # ------------------------
    data_dir = config["paths"]["raw"]
    out_dir = config["paths"]["processed"]

    num_visualize = config["preprocess"].get("num_visualize", 20)
    save_masks_fmt = config["preprocess"].get("save_masks", ["png"])
    make_splits_flag = config["preprocess"].get("make_splits", True)
    n_folds = config["preprocess"].get("n_folds", 5)
    mini_ratio = config["preprocess"].get("mini_ratio", None)

    # Override dosya pathleri
    train_file = config["preprocess"].get("train_file", os.path.join(out_dir, "splits/train.txt"))
    val_file = config["preprocess"].get("val_file", os.path.join(out_dir, "splits/val.txt"))
    train_file_mini = config["preprocess"].get("train_file_mini", os.path.join(out_dir, "splits/train_mini.txt"))
    val_file_mini = config["preprocess"].get("val_file_mini", os.path.join(out_dir, "splits/val_mini.txt"))

    # ------------------------
    # Yol tanımları
    # ------------------------
    raw_train_dir = os.path.join(data_dir, "train_images")
    train_csv = os.path.join(data_dir, "train.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv bulunamadı: {train_csv}")
    if not os.path.isdir(raw_train_dir):
        raise FileNotFoundError(f"train_images klasörü yok: {raw_train_dir}")

    # Çıkış klasörleri
    masks_png_dir = os.path.join(out_dir, "masks_png")
    overlays_dir = os.path.join(out_dir, "overlays")
    splits_dir = os.path.join(out_dir, "splits")
    reports_dir = os.path.join(out_dir, "reports")

    for p in [out_dir, overlays_dir, reports_dir, masks_png_dir, splits_dir]:
        ensure_dir(p)
    for c in [1, 2, 3, 4]:
        ensure_dir(os.path.join(masks_png_dir, f"class{c}"))

    # ------------------------
    # CSV oku
    # ------------------------
    df = pd.read_csv(train_csv)
    if "ImageId_ClassId" in df.columns:
        tmp = df["ImageId_ClassId"].str.split("_", expand=True)
        df["ImageId"] = tmp[0]
        df["ClassId"] = tmp[1].astype(int)
    elif "ImageId" not in df.columns or "ClassId" not in df.columns:
        raise ValueError("CSV formatı beklenen sütunlara sahip değil.")

    df["EncodedPixels"] = df["EncodedPixels"].fillna("")
    df["ClassId"] = df["ClassId"].astype(int)

    all_image_ids = df["ImageId"].unique().tolist()

    # ------------------------
    # Örnek overlay üret & Rapor için veri topla
    # ------------------------
    np.random.seed(42)
    sample_ids = np.random.choice(all_image_ids, size=min(num_visualize, len(all_image_ids)), replace=False)

    suspicious_full_masks = []
    overlap_issue_count = 0
    coverage_hist = {c: [] for c in [1, 2, 3, 4]}

    print(f"[INFO] {len(sample_ids)} görsel için overlay üretiliyor ve veri analizi yapılıyor...")
    for image_id in tqdm(sample_ids):
        img_path = os.path.join(raw_train_dir, image_id)
        img = cv2.imread(img_path)
        if img is None:
            continue

        masks = []
        for c in [1, 2, 3, 4]:
            rle_series = df.loc[(df.ImageId == image_id) & (df.ClassId == c), "EncodedPixels"]
            rle = rle_series.values[0] if len(rle_series) > 0 else ""
            mask_c = rle_decode(rle, shape=(H, W))
            masks.append(mask_c)
            
            # ✅ Kapsam oranını (coverage) hesapla
            if mask_c.sum() > 0:
                coverage = mask_c.sum() / (H * W)
                coverage_hist[c].append(float(coverage))

        masks = np.stack(masks, axis=-1)

        if (masks.sum(axis=-1) > 1).any():
            overlap_issue_count += 1

        for c in range(4):
            if masks[..., c].sum() == H * W:
                suspicious_full_masks.append(f"{image_id}_class{c+1}")

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        out = img.copy()
        for c in range(4):
            if masks[..., c].any():
                out = overlay_mask_on_image(out, masks[..., c], color=colors[c], alpha=0.35)
        cv2.imwrite(os.path.join(overlays_dir, f"{os.path.splitext(image_id)[0]}_overlay.jpg"), out)

    # ------------------------
    # Maskeleri kaydet
    # ------------------------
    if "png" in save_masks_fmt:
        print("[INFO] PNG maskeler kaydediliyor...")
        for image_id in tqdm(all_image_ids):
            save_masks_png(image_id, df, masks_png_dir, shape=(H, W))

    # ------------------------
    # Train/Val split
    # ------------------------
    if make_splits_flag:
        print("[INFO] Train/Val split oluşturuluyor...")
        any_defect = (
            df.pivot_table(index="ImageId", columns="ClassId", values="EncodedPixels",
                            aggfunc=lambda x: any([not is_empty_rle(v) for v in x]))
            .fillna(False)
            .any(axis=1)
            .astype(int)
        )

        # Full split
        make_train_val_split(all_image_ids, any_defect, splits_dir, n_folds=n_folds, val_fold=0,
                             train_file=train_file, val_file=val_file)

        # Mini split
        if mini_ratio:
            print(f"[INFO] Mini split oluşturuluyor (ratio={mini_ratio})...")
            with open(train_file, "r") as f:
                train_ids = [line.strip() for line in f.readlines()]
            with open(val_file, "r") as f:
                val_ids = [line.strip() for line in f.readlines()]

            np.random.seed(42)
            mini_train = np.random.choice(train_ids, size=max(1, int(len(train_ids) * mini_ratio)), replace=False)
            mini_val = np.random.choice(val_ids, size=max(1, int(len(val_ids) * mini_ratio)), replace=False)

            with open(train_file_mini, "w") as f:
                f.write("\n".join(mini_train))
            with open(val_file_mini, "w") as f:
                f.write("\n".join(mini_val))

    # ------------------------
    # Rapor
    # ------------------------
    report = generate_report(df, coverage_hist, overlap_issue_count, suspicious_full_masks, reports_dir)

    print("\n===== ÖZET RAPOR =====")
    print(json.dumps(report, indent=2))
    print("======================")


if __name__ == "__main__":
    main()