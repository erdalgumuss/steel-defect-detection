#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train/Val split üretir (stratified by any_defect).
Çıktılar:
  data/processed/splits/train.txt
  data/processed/splits/val.txt
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_empty_rle(x) -> bool:
    return not isinstance(x, str) or x.strip() == ""

def main(args):
    H, W = 256, 1600  # dataset sabit boyut, aslında burada gerek yok

    # CSV oku
    df = pd.read_csv(args.train_csv)

    # Beklenen sütun kontrolü
    if "ImageId" not in df.columns or "ClassId" not in df.columns:
        if "ImageId_ClassId" in df.columns and "EncodedPixels" in df.columns:
            tmp = df["ImageId_ClassId"].str.split("_", expand=True)
            df["ImageId"] = tmp[0]
            df["ClassId"] = tmp[1].astype(int)
        else:
            raise ValueError("CSV formatı beklenenden farklı: ImageId, ClassId, EncodedPixels gerekli.")

    # any_defect bilgisi çıkar
    any_defect_per_img = df.pivot_table(
        index="ImageId",
        columns="ClassId",
        values="EncodedPixels",
        aggfunc=lambda x: any([not is_empty_rle(v) for v in x])
    )
    any_defect_per_img = any_defect_per_img.fillna(False)
    any_defect = any_defect_per_img.any(axis=1).astype(int)

    img_df = any_defect.to_frame(name="any_defect").reset_index()

    # Stratified split (default: 5-fold, val fold = 0)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_ids = np.zeros(len(img_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(img_df["ImageId"], img_df["any_defect"])):
        fold_ids[val_idx] = fold
    img_df["fold"] = fold_ids

    val_fold = args.val_fold
    train_list = img_df[img_df["fold"] != val_fold]["ImageId"].tolist()
    val_list   = img_df[img_df["fold"] == val_fold]["ImageId"].tolist()

    ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(args.out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_list))

    print(f"[INFO] Train/Val split hazır: {args.out_dir}")
    print(f" - Train set: {len(train_list)}")
    print(f" - Val set:   {len(val_list)}")
    print(f" - Val fold:  {val_fold}")

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/raw/train.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed/splits")
    parser.add_argument("--n_folds", type=int, default=5, help="Stratified K-fold sayısı")
    parser.add_argument("--val_fold", type=int, default=0, help="Validation için hangi fold seçilecek")
    args = parser.parse_args()
    main(args)
