"""Reject obvious non-leaf uploads before the classifier runs.

The crop model only knows 40 disease classes. On photos of houses, terminals, or
documents it still picks a "best guess" with very high softmax scores. Rules here
catch those cases using simple color and contrast checks — not perfect, but it
blocks the worst false positives without retraining.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def looks_like_crop_leaf_photo(image: Image.Image) -> bool:
    """
    Return True only if the image plausibly contains plant material.

    Designed to reject: terminal screenshots, UI/text, many buildings, random objects.
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

    green = (g > r + 10) & (g > b + 10) & (g >= 50)
    green_frac = float(green.mean())

    # Terminal / IDE: dark console, light text, or mostly white document backgrounds
    if dark_frac > 0.12 and bright_frac > 0.03:
        return False
    if bright_frac > 0.42 and mid_frac < 0.24:
        return False
    if bright_frac > 0.35 and mean_chroma < 22:
        return False

    # Mostly grayscale (commands, B&W scans, some renders)
    if mean_chroma < 16:
        return False

    # Large bright flat background with almost no green (documents, many renders)
    if bright_frac > 0.32 and green_frac < 0.04:
        return False

    # Syntax-highlight green on white terminals is sparse — not a leaf photo
    if bright_frac > 0.35 and green_frac < 0.12:
        return False

    # Strong UI blues (screens, app chrome) without foliage
    ui_blue = (b > r + 25) & (b > g + 15) & (b > 90)
    if float(ui_blue.mean()) > 0.14 and green_frac < 0.05:
        return False

    # Need clear plant-like color: healthy green OR diseased/brown organic tones
    if green_frac >= 0.10 and mid_frac >= 0.22:
        return True
    if green_frac >= 0.055 and bright_frac < 0.35:
        return True

    organic = (chroma >= 22) & (lum >= 40) & (lum <= 210)
    organic = organic & ~((b > r + 28) & (b > g + 20))  # drop screen-blue
    organic_frac = float(organic.mean())

    # Brown/yellow spots on leaves, soil-toned diseased foliage
    if organic_frac >= 0.28 and green_frac >= 0.02 and mid_frac >= 0.25:
        return True

    return False
