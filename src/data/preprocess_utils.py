#src/data/preprocess_utils.py
import os
import json
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold

# ---------------------------
# RLE Encode/Decode
# ---------------------------

def rle_decode(mask_rle: str, shape=(256, 1600)) -> np.ndarray:
    """RLE string -> binary mask"""
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.strip().split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Binary mask -> RLE string (Fortran order)"""
    pixels = mask.flatten(order="F")
    pads = np.pad(pixels, (1, 1), mode="constant")
    changes = np.where(pads[1:] != pads[:-1])[0] + 1
    starts = changes[::2]
    ends = changes[1::2]
    lengths = ends - starts
    if len(starts) == 0:
        return ""
    return " ".join([f"{s} {l}" for s, l in zip(starts, lengths)])


# ---------------------------
# Yardımcılar
# ---------------------------

def ensure_dir(path: str):
    """Klasör varsa geç, yoksa oluştur"""
    os.makedirs(path, exist_ok=True)


def is_empty_rle(x) -> bool:
    """RLE boş mu kontrolü"""
    return not isinstance(x, str) or x.strip() == ""


def overlay_mask_on_image(image_bgr: np.ndarray, mask: np.ndarray,
                          color=(0, 0, 255), alpha=0.4) -> np.ndarray:
    """Overlay a binary mask onto an image"""
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


# ---------------------------
# Mask & Split Utils
# ---------------------------

def save_masks_png(image_id: str,
                   df: pd.DataFrame,
                   out_dir: str,
                   shape: Tuple[int, int] = (256, 1600)):
    """
    RLE'den maskeleri çıkar ve class bazlı PNG olarak kaydet
    """
    base = os.path.splitext(image_id)[0]
    for c in [1, 2, 3, 4]:
        rle = df.loc[(df.ImageId == image_id) & (df.ClassId == c), "EncodedPixels"]
        rle = rle.values[0] if len(rle) else ""
        mask_c = rle_decode(rle, shape=shape)
        png_path = os.path.join(out_dir, f"class{c}", f"{base}_c{c}.png")
        ensure_dir(os.path.dirname(png_path))
        cv2.imwrite(png_path, (mask_c * 255).astype(np.uint8))


def make_train_val_split(image_ids: List[str],
                         any_defect: pd.Series,
                         out_dir: str,
                         n_folds: int = 5,
                         val_fold: int = 0,
                         train_file: str = None,
                         val_file: str = None) -> Dict[str, List[str]]:
    """
    Stratified K-Fold split (any_defect üzerinden).
    Eğer config'te özel train/val dosya yolları verilmişse oraya yazar.
    """
    ensure_dir(out_dir)
    img_df = any_defect.to_frame(name="any_defect").reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_ids = np.zeros(len(img_df), dtype=np.int64)

    for fold, (_, val_idx) in enumerate(skf.split(img_df["ImageId"], img_df["any_defect"])):
        fold_ids[val_idx] = fold
    img_df["fold"] = fold_ids

    train_list = img_df[img_df["fold"] != val_fold]["ImageId"].tolist()
    val_list = img_df[img_df["fold"] == val_fold]["ImageId"].tolist()

    # Config’ten gelen path varsa onu kullan
    train_path = train_file if train_file else os.path.join(out_dir, "train.txt")
    val_path = val_file if val_file else os.path.join(out_dir, "val.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_list))
    with open(val_path, "w") as f:
        f.write("\n".join(val_list))

    return {"train": train_list, "val": val_list}

# ---------------------------
# Raporlama
# ---------------------------

def generate_report(df: pd.DataFrame,
                    coverage_hist: Dict[int, List[float]],
                    overlap_issue_count: int,
                    suspicious_full_masks: List[str],
                    out_dir: str):
    """
    JSON rapor üretir: class coverage, overlap, suspicious masks vs.
    """
    report = {
        "num_rows_csv": int(len(df)),
        "num_unique_images": int(df["ImageId"].nunique()),
        "class_positive_counts": {
            c: int(df[df["ClassId"] == c]["EncodedPixels"].apply(lambda x: not is_empty_rle(x)).sum())
            for c in [1, 2, 3, 4]
        },
        "overlap_issue_in_sampled": int(overlap_issue_count),
        "suspicious_full_masks": suspicious_full_masks,
    }

    # coverage özet
    for c in [1, 2, 3, 4]:
        arr = np.array(coverage_hist[c]) if len(coverage_hist[c]) else np.array([0.0])
        report[f"class_{c}_coverage_mean"] = float(arr.mean())
        report[f"class_{c}_coverage_median"] = float(np.median(arr))

    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "preprocess_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report
