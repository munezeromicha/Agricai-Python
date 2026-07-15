"""Pure numpy/PIL image-quality checks — blur, resolution, brightness.

No ML model, no OpenCV. Runs before the Roboflow call so obviously bad photos
(out of focus, too dark/bright, thumbnails) are rejected without spending an API call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from app.config import Settings

QualityIssueKind = Literal["too_small", "too_dark", "too_bright", "blurry"]


@dataclass(frozen=True)
class QualityIssue:
    kind: QualityIssueKind
    detail: str


def _to_gray_array(image: Image.Image, max_edge: int = 512) -> np.ndarray:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the discrete Laplacian — low value = low high-frequency detail = blurry."""
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
    )
    return float(lap.var())


def blur_score(image: Image.Image) -> float:
    return laplacian_variance(_to_gray_array(image))


def is_blurry(image: Image.Image, *, threshold: float) -> bool:
    return blur_score(image) < threshold


def brightness_mean(image: Image.Image) -> float:
    gray = _to_gray_array(image, max_edge=128)
    return float(gray.mean())


def is_too_dark(image: Image.Image, *, min_mean: float) -> bool:
    return brightness_mean(image) < min_mean


def is_too_bright(image: Image.Image, *, max_mean: float) -> bool:
    return brightness_mean(image) > max_mean


def resolution_ok(image: Image.Image, *, min_edge: int) -> bool:
    w, h = image.size
    return min(w, h) >= min_edge


def assess_image_quality(image: Image.Image, settings: Settings) -> QualityIssue | None:
    """Cheapest/most-decisive checks first: resolution -> brightness -> blur."""
    if not resolution_ok(image, min_edge=settings.image_quality_min_edge_px):
        w, h = image.size
        return QualityIssue("too_small", f"image {w}x{h} below min edge {settings.image_quality_min_edge_px}px")

    mean_luma = brightness_mean(image)
    if mean_luma < settings.image_quality_min_mean_luma:
        return QualityIssue("too_dark", f"mean luma {mean_luma:.1f} < {settings.image_quality_min_mean_luma}")
    if mean_luma > settings.image_quality_max_mean_luma:
        return QualityIssue("too_bright", f"mean luma {mean_luma:.1f} > {settings.image_quality_max_mean_luma}")

    score = blur_score(image)
    if score < settings.image_quality_blur_variance_threshold:
        return QualityIssue("blurry", f"blur variance {score:.1f} < {settings.image_quality_blur_variance_threshold}")

    return None
