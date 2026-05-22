"""Reject obvious non-leaf uploads before the classifier runs.

The crop model only knows trained disease classes. On photos of houses, terminals, or
documents it still picks a "best guess" with high softmax scores. Rules here catch
those cases using simple color checks — tuned to allow diseased/brown foliage.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def looks_like_crop_leaf_photo(image: Image.Image) -> bool:
    """
    Return True only if the image plausibly contains plant material.

    Designed to reject: terminal screenshots, UI/text, many buildings, random objects.
    Allows: brown/rust/spotted leaves, dry foliage, partial green.
    """
    rgb = image.convert("RGB")
    small = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    px = np.asarray(small, dtype=np.float32)
    r = px[..., 0]
    g = px[..., 1]
    b = px[..., 2]

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = np.max(px, axis=-1) - np.min(px, axis=-1)

    dark_frac = float((lum < 45).mean())
    bright_frac = float((lum > 215).mean())
    mid_frac = float(((lum >= 60) & (lum <= 200)).mean())
    mean_chroma = float(chroma.mean())

    green = (g > r + 8) & (g > b + 8) & (g >= 45)
    green_frac = float(green.mean())

    # Terminal / IDE: dark console + light text
    if dark_frac > 0.12 and bright_frac > 0.03:
        return False
    if bright_frac > 0.48 and mid_frac < 0.20:
        return False
    if bright_frac > 0.40 and mean_chroma < 20:
        return False

    # Mostly grayscale (commands, B&W scans)
    if mean_chroma < 14:
        return False

    # Large bright flat background with almost no plant color
    if bright_frac > 0.38 and green_frac < 0.03:
        return False

    # Syntax-highlight green on white terminals
    if bright_frac > 0.38 and green_frac < 0.10:
        return False

    # Strong UI blues without foliage
    ui_blue = (b > r + 25) & (b > g + 15) & (b > 90)
    if float(ui_blue.mean()) > 0.16 and green_frac < 0.04:
        return False

    # Healthy / green foliage
    if green_frac >= 0.08 and mid_frac >= 0.20:
        return True
    if green_frac >= 0.045 and bright_frac < 0.42:
        return True

    # Diseased, rust, brown, yellow — organic tones without much green
    organic = (chroma >= 18) & (lum >= 35) & (lum <= 220)
    organic = organic & ~((b > r + 28) & (b > g + 20))
    organic_frac = float(organic.mean())

    # Brown/yellow rust spots, blighted tissue, soil-toned leaves
    warm = (r >= g - 5) & (r >= 40) & (lum >= 40) & (lum <= 200) & (chroma >= 15)
    warm_frac = float(warm.mean())

    if organic_frac >= 0.22 and mid_frac >= 0.22:
        return True
    if warm_frac >= 0.18 and mid_frac >= 0.20:
        return True
    if organic_frac >= 0.16 and green_frac >= 0.02:
        return True

    return False
