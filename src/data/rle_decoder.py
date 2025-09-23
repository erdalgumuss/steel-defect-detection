# src/data/rle_decoder.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def rle_to_mask(rle_string: str | float | None, shape: Tuple[int, int] = (256, 1600), *, order: str = "F") -> np.ndarray:
    """
    Decode a single RLE string into a binary mask.

    Args:
        rle_string: RLE encoded string ("start length start length ..."). May be NaN/None for empty mask.
        shape: (H, W) of the output mask.
        order: Memory order used by the dataset's RLE convention. Most steel datasets use Fortran ('F').

    Returns:
        np.ndarray of shape (H, W), dtype=uint8, with ones on defect pixels.
    """
    if rle_string is None or (isinstance(rle_string, float) and np.isnan(rle_string)) or (isinstance(rle_string, str) and rle_string.strip() == ""):
        return np.zeros(shape, dtype=np.uint8)

    s = np.asarray(rle_string.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths

    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    for st, en in zip(starts, ends):
        mask[st:en] = 1

    # Important: many RLEs for steel defect datasets assume column-major (Fortran) order
    return mask.reshape(shape, order=order)


def build_multilabel_mask(rows_for_image: pd.DataFrame, shape: Tuple[int, int] = (256, 1600), num_classes: int = 4,
                           *, order: str = "F") -> np.ndarray:
    """
    Build a (C, H, W) multilabel mask for a single image given its per-class rows.

    Expects rows_for_image columns to include: ['ImageId', 'ClassId', 'EncodedPixels'].
    ClassId is assumed to be in {1, 2, 3, 4}.
    """
    out = np.zeros((num_classes, shape[0], shape[1]), dtype=np.uint8)
    if rows_for_image is None or len(rows_for_image) == 0:
        return out

    for _, row in rows_for_image.iterrows():
        if pd.isna(row["ClassId"]):   # 👈 NaN ise devam et
            continue
        cls = int(row["ClassId"]) - 1
        if 0 <= cls < num_classes:
            out[cls] |= rle_to_mask(row["EncodedPixels"], shape=shape, order=order)
    return out



# --------- Quick self-tests (can be executed in Notebook context) ---------
if __name__ == "__main__":
    # Empty/NaN case
    m = rle_to_mask(np.nan, (4, 5))
    assert m.sum() == 0 and m.shape == (4, 5)

    # Simple run: pixels 3..5 (1-based: start=3, len=3) in a 3x5 reshaped F-order
    test = rle_to_mask("3 3", (3, 5))
    assert test.shape == (3, 5)
    # Just a sanity check that some pixels are on
    assert test.sum() == 3
    print("rle_decoder self-tests passed ✅")

