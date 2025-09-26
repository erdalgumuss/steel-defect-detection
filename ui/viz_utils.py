# src/ui/viz_utils.py
import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_distinct_colors(n, seed: int = 42):
    """
    Generate n visually distinct colors using matplotlib's colormap.
    Deterministic with given seed.
    """
    np.random.seed(seed)
    cmap = plt.cm.get_cmap("tab10", n)  # up to 10 distinct colors
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def overlay_masks(image_np, masks, alpha=0.4, colors=None):
    """
    Overlay segmentation masks on an image.

    Args:
        image_np: (H, W, 3) RGB image (np.uint8)
        masks: (C, H, W) binary masks
        alpha: mask transparency (0..1)
        colors: list of (R,G,B) tuples; if None, auto-generate
    """
    overlay = image_np.copy()
    C, H, W = masks.shape

    if colors is None or len(colors) < C:
        colors = generate_distinct_colors(C)

    for i in range(C):
        m = (masks[i] > 0).astype(np.uint8)
        if m.sum() == 0:
            continue
        colored_mask = np.zeros_like(image_np)
        for ch in range(3):
            colored_mask[..., ch] = m * colors[i][ch]
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

    return overlay
