"""Extract the primary leaf region before disease classification.

Field photos often include soil, multiple leaves, or extra background. Cropping to the
largest leaf-colored blob improves both plant-guard scoring and classifier accuracy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.inference.plant_guard import _leaf_color_masks


def _largest_blob_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    if not mask.any():
        return None

    visited = np.zeros_like(mask, dtype=bool)
    best_box: tuple[int, int, int, int] | None = None
    best_size = 0

    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        y_min = y_max = int(y0)
        x_min = x_max = int(x0)
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if size > best_size:
            best_size = size
            best_box = (x_min, y_min, x_max + 1, y_max + 1)
    return best_box


def extract_primary_leaf(
    image: Image.Image,
    *,
    min_blob_frac: float = 0.025,
    padding_frac: float = 0.10,
) -> tuple[Image.Image, bool]:
    """
    Crop to the largest leaf-colored region in the image.

    Returns (cropped_image, was_cropped).
    If no sufficiently large leaf blob is found, returns the original image.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < 32 or h < 32:
        return rgb, False

    analysis = rgb.resize((384, 384), Image.Resampling.BILINEAR)
    px = np.asarray(analysis, dtype=np.float32)
    leaf_mask = _leaf_color_masks(px)["leaf"]

    if float(leaf_mask.mean()) < min_blob_frac:
        return rgb, False

    bbox = _largest_blob_bbox(leaf_mask)
    if bbox is None:
        return rgb, False

    ax0, ay0, ax1, ay1 = bbox
    blob_area = (ax1 - ax0) * (ay1 - ay0)
    if blob_area / float(leaf_mask.size) < min_blob_frac:
        return rgb, False

    # Map analysis bbox back to original coordinates
    scale_x = w / 384.0
    scale_y = h / 384.0
    x0 = int(ax0 * scale_x)
    y0 = int(ay0 * scale_y)
    x1 = int(ax1 * scale_x)
    y1 = int(ay1 * scale_y)

    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    # Square crop around leaf center for model input consistency
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(x1 - x0, y1 - y0)
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    if x1 - x0 < 48 or y1 - y0 < 48:
        return rgb, False

    return rgb.crop((x0, y0, x1, y1)), True
