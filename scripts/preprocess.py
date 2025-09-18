#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess Severstal Steel Defect Detection
- train.csv okur, RLE -> mask
- Tutarlılık kontrolleri (4 satır, boyutlar, NaN, overlap, coverage)
- Sınıf dağılımı ve özet rapor
- Örnek overlay görseller
- (Opsiyon) maskeleri .npz ve/veya sınıf başına PNG olarak kaydeder
- Train/Val split üretir (basit, stratified by 'any_defect')

Koşu örneği:
python scripts/preprocess.py \
  --data_dir data/raw \
  --out_dir data/processed \
  --num_visualize 20 \
  --save_masks npz png \
  --make_splits
"""

import os
import argparse
import json
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------
# RLE yardımcıları
# ---------------------------

def rle_decode(mask_rle: str, shape=(256, 1600)) -> np.ndarray:
    """
    RLE 'start length start length ...' -> binary mask
    Piksel sıralaması: sütun major (Fortran), üstten aşağı, soldan sağa
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.strip().split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def rle_encode(mask: np.ndarray) -> str:
    """
    Binary mask -> RLE string (Fortran order)
    İleride submission için lazım olacak; şimdiden ekliyoruz.
    """
    pixels = mask.flatten(order='F')
    # pad zeros at ends to catch runs
    pads = np.pad(pixels, (1, 1), mode='constant')
    changes = np.where(pads[1:] != pads[:-1])[0] + 1
    starts = changes[::2]
    ends = changes[1::2]
    lengths = ends - starts
    if len(starts) == 0:
        return ""  # boş maske
    return " ".join([f"{s} {l}" for s, l in zip(starts, lengths)])


# ---------------------------
# Yardımcılar
# ---------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_empty_rle(x) -> bool:
    return not isinstance(x, str) or (isinstance(x, str) and x.strip() == "")

def overlay_mask_on_image(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4):
    """
    mask: 0/1, HxW
    color: BGR
    """
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


# ---------------------------
# Ana iş akışı
# ---------------------------

def main(args):
    H, W = 256, 1600  # Severstal sabit boyut

    # Yol kurulum
    data_dir = args.data_dir
    out_dir = args.out_dir
    raw_train_dir = os.path.join(data_dir, "train_images")
    raw_test_dir  = os.path.join(data_dir, "test_images")
    train_csv     = os.path.join(data_dir, "train.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv bulunamadı: {train_csv}")
    if not os.path.isdir(raw_train_dir):
        raise FileNotFoundError(f"train_images klasörü yok: {raw_train_dir}")

    # Çıkış klasörleri
    masks_npz_dir   = os.path.join(out_dir, "masks_npz")
    masks_png_dir   = os.path.join(out_dir, "masks_png")
    overlays_dir    = os.path.join(out_dir, "overlays")
    splits_dir      = os.path.join(out_dir, "splits")
    reports_dir     = os.path.join(out_dir, "reports")
    for p in [out_dir, overlays_dir, reports_dir]:
        ensure_dir(p)
    if "npz" in args.save_masks:
        ensure_dir(masks_npz_dir)
    if "png" in args.save_masks:
        ensure_dir(masks_png_dir)
        for c in [1,2,3,4]:
            ensure_dir(os.path.join(masks_png_dir, f"class{c}"))

    # CSV oku
    df = pd.read_csv(train_csv)
    # Beklenen sütun kontrolü (bazı kopyalarda ImageId_ClassId olabilir)
    if "ImageId" not in df.columns or "ClassId" not in df.columns:
        # muhtemel alternatif formatı dönüştürme
        if "ImageId_ClassId" in df.columns and "EncodedPixels" in df.columns:
            tmp = df["ImageId_ClassId"].str.split("_", expand=True)
            df["ImageId"] = tmp[0]
            df["ClassId"] = tmp[1].astype(int)
        else:
            raise ValueError("Beklenen sütunlar yok. Gerekli sütunlar: ImageId, ClassId, EncodedPixels")

    # Temizle
    df["EncodedPixels"] = df["EncodedPixels"].fillna("")
    df["ClassId"] = df["ClassId"].astype(int)

    # 1) her görüntü için 4 kayıt var mı?
    counts_per_image = df.groupby("ImageId")["ClassId"].nunique()
    missing_4 = (counts_per_image != 4).sum()
    # 2) Görseller dosyada mevcut mu?
    all_image_ids = df["ImageId"].unique().tolist()
    missing_files = [img for img in all_image_ids if not os.path.exists(os.path.join(raw_train_dir, img))]
    # 3) sınıf sayımı
    cls_pos_counts = {}
    for c in [1,2,3,4]:
        m = df[df["ClassId"]==c]["EncodedPixels"].apply(lambda x: not is_empty_rle(x)).sum()
        cls_pos_counts[c] = int(m)

    # 4) any_defect oranı
    any_defect_per_img = df.pivot_table(index="ImageId",
                                        columns="ClassId",
                                        values="EncodedPixels",
                                        aggfunc=lambda x: any([not is_empty_rle(v) for v in x]))
    any_defect_per_img = any_defect_per_img.fillna(False)
    any_defect = any_defect_per_img.any(axis=1).astype(int)
    no_defect_ratio = float((any_defect == 0).mean())

    # 5) örnek görselleştirme
    np.random.seed(42)
    sample_ids = np.random.choice(all_image_ids, size=min(args.num_visualize, len(all_image_ids)), replace=False)

    # istatistikler
    suspicious_full_masks = []
    overlap_issue_count = 0
    coverage_hist = defaultdict(list)

    print(f"[INFO] Görselleştirme ve kontroller başlıyor ({len(sample_ids)} örnek)...")
    for image_id in tqdm(sample_ids):
        img_path = os.path.join(raw_train_dir, image_id)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Okunamadı: {img_path}")
            continue

        # 4 sınıf maskeyi oluştur
        masks = []
        for c in [1,2,3,4]:
            rle = df.loc[(df.ImageId==image_id) & (df.ClassId==c), "EncodedPixels"]
            rle = rle.values[0] if len(rle) else ""
            mask_c = rle_decode(rle, shape=(H, W))
            masks.append(mask_c)
            # coverage toplayalım (yüzde)
            coverage = mask_c.sum() / (H*W)
            coverage_hist[c].append(coverage)

        masks = np.stack(masks, axis=-1)  # HxWx4

        # Overlap kontrol (aynı pikselde birden fazla sınıf?)
        if (masks.sum(axis=-1) > 1).any():
            overlap_issue_count += 1

        # "tam dolu" (ör. 1 409600) şüphelisini işaretle
        for c in [0,1,2,3]:
            if masks[..., c].sum() == H*W:
                suspicious_full_masks.append(f"{image_id}_class{c+1}")

        # overlay üret (renkler: BGR)
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
        out = img.copy()
        for c in [0,1,2,3]:
            if masks[...,c].any():
                out = overlay_mask_on_image(out, masks[...,c], color=colors[c], alpha=0.35)
        cv2.imwrite(os.path.join(overlays_dir, f"{os.path.splitext(image_id)[0]}_overlay.jpg"), out)

    # (Opsiyon) tüm maskeleri kaydet
    if args.save_masks:
        print("[INFO] Maskeleri kaydetme başlıyor (tüm görüntüler için)...")
        for image_id in tqdm(all_image_ids):
            # img var mı diye yine kontrol et
            img_path = os.path.join(raw_train_dir, image_id)
            if not os.path.exists(img_path):
                continue

            masks = []
            for c in [1,2,3,4]:
                rle = df.loc[(df.ImageId==image_id) & (df.ClassId==c), "EncodedPixels"]
                rle = rle.values[0] if len(rle) else ""
                mask_c = rle_decode(rle, shape=(H, W))
                masks.append(mask_c)
            masks = np.stack(masks, axis=-1)  # HxWx4

            base = os.path.splitext(image_id)[0]

            if "npz" in args.save_masks:
                np.savez_compressed(os.path.join(masks_npz_dir, f"{base}.npz"), mask=masks.astype(np.uint8))

            if "png" in args.save_masks:
                for c in [1,2,3,4]:
                    png_path = os.path.join(masks_png_dir, f"class{c}", f"{base}_c{c}.png")
                    cv2.imwrite(png_path, (masks[...,c-1]*255).astype(np.uint8))

    # Basit stratified split (any_defect)
    if args.make_splits:
        ensure_dir(splits_dir)
        # any_defect'a göre stratify
        img_df = any_defect.to_frame(name="any_defect").reset_index()
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        fold_ids = np.zeros(len(img_df), dtype=np.int64)
        for fold, (_, val_idx) in enumerate(skf.split(img_df["ImageId"], img_df["any_defect"])):
            fold_ids[val_idx] = fold
        img_df["fold"] = fold_ids
        # 1 fold'u val, kalanı train gibi basit kullanım
        val_fold = 0
        train_list = img_df[img_df["fold"] != val_fold]["ImageId"].tolist()
        val_list   = img_df[img_df["fold"] == val_fold]["ImageId"].tolist()

        with open(os.path.join(splits_dir, "train.txt"), "w") as f:
            f.write("\n".join(train_list))
        with open(os.path.join(splits_dir, "val.txt"), "w") as f:
            f.write("\n".join(val_list))

    # Rapor
    report = {
        "num_rows_csv": int(len(df)),
        "num_unique_images": int(len(all_image_ids)),
        "images_missing_4_rows": int(missing_4),
        "missing_image_files": missing_files,
        "class_positive_counts": cls_pos_counts,
        "no_defect_ratio": no_defect_ratio,
        "overlap_issue_in_sampled": int(overlap_issue_count),
        "suspicious_full_masks_in_sampled": suspicious_full_masks,
        "notes": [
            "overlap_issue_in_sampled > 0 ise, sınıflar arası çakışma var demektir (etiket problemi olabilir).",
            "suspicious_full_masks (tam dolu maske) tipik değil; örnekleri gözle kontrol edin."
        ]
    }

    # coverage özet (medyan/ortalama)
    cov_summary = {}
    for c in [1,2,3,4]:
        arr = np.array(coverage_hist[c]) if len(coverage_hist[c]) else np.array([0.0])
        cov_summary[f"class_{c}_coverage_mean"] = float(arr.mean())
        cov_summary[f"class_{c}_coverage_median"] = float(np.median(arr))
    report.update(cov_summary)

    ensure_dir(reports_dir)
    with open(os.path.join(reports_dir, "preprocess_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== ÖZET RAPOR =====")
    for k,v in report.items():
        print(f"{k}: {v}")
    print(f"\nOverlay görseller: {overlays_dir}")
    if args.save_masks:
        print(f"Maskeler: npz={ 'npz' in args.save_masks }  png={ 'png' in args.save_masks }")
    if args.make_splits:
        print(f"Split dosyaları: {splits_dir}/train.txt, {splits_dir}/val.txt")
    print("======================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Kaggle orijinal veri klasörü")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Çıktı klasörü")
    parser.add_argument("--num_visualize", type=int, default=20, help="Kaç örnek overlay üretilecek")
    parser.add_argument("--save_masks", nargs="*", default=[], choices=["npz","png"], help="Maskeleri kaydet (npz/png)")
    parser.add_argument("--make_splits", action="store_true", help="Train/Val split üret")
    parser.add_argument("--n_folds", type=int, default=5, help="Stratified K-fold sayısı (any_defect)")
    args = parser.parse_args()
    main(args)
