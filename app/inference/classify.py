"""Shared classification logic — ONNX inference, TTA, and confidence gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from app.config import Settings

RejectionReason = Literal[
    "plant_guard",
    "not_tomato",
    "low_confidence",
    "low_margin",
    "class_count_mismatch",
]


@dataclass(frozen=True)
class ClassScore:
    class_id: str
    disease_name: str
    confidence_pct: float


@dataclass(frozen=True)
class ClassifyDetails:
    """Raw classifier output before mapping to knowledge-base copy."""

    probs: np.ndarray
    top_class_id: str | None
    uncertain: bool
    rejection_reason: RejectionReason | None
    top_confidence: float
    margin: float
    alternatives: list[ClassScore]
    plant_guard_blocked: bool = False
    tomato_gate_score: float | None = None
    tomato_gate_soft_pass: bool = False


def channel_dim_index(shape: tuple) -> int | None:
    if len(shape) != 4:
        return None
    for i in (1, 2, 3):
        dim = shape[i]
        if dim == 3 or dim == "3":
            return i
    return None


def infer_onnx_layout(input_shape: tuple) -> str:
    ch = channel_dim_index(input_shape)
    if ch == 1:
        return "nchw"
    if ch == 3:
        return "nhwc"
    return "nhwc"


def preprocess_imagenet(image: Image.Image, size: int, layout: str = "nhwc") -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    if layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)


def normalize_logits(raw: np.ndarray) -> np.ndarray:
    """
    ONNX outputs vary: (num_classes,), (1, num_classes), (num_classes, 1).
    Never use raw[0] on a 1D array — that returns a single scalar, not the vector.
    """
    arr = np.asarray(raw, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("Model output is a scalar; expected a class logits vector.")
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[1] == 1:
            return arr[:, 0]
    return arr.reshape(-1)


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """1D logits -> 1D probabilities."""
    vec = normalize_logits(logits)
    return softmax(vec.reshape(1, -1))[0]


def tta_image_variants(image: Image.Image) -> list[Image.Image]:
    """Views that mimic common field-photo variation (angle, framing, lighting)."""
    rgb = image.convert("RGB")
    variants: list[Image.Image] = [rgb]
    variants.append(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

    w, h = rgb.size
    for inset in (0.06, 0.12):
        dx = int(w * inset)
        dy = int(h * inset)
        if w > 2 * dx and h > 2 * dy:
            variants.append(rgb.crop((dx, dy, w - dx, h - dy)))
    return variants


def is_overconfident_nonleaf(
    probs: np.ndarray,
    leaf_score: float,
    *,
    min_leaf_score: float,
    confidence_trigger: float,
) -> bool:
    """Catch OOD photos where softmax is wrongly very confident (people, landscapes, etc.)."""
    if probs.size == 0:
        return False
    top_p = float(np.max(probs))
    return top_p >= confidence_trigger and leaf_score < min_leaf_score


def prediction_is_uncertain(
    probs: np.ndarray,
    *,
    min_confidence: float,
    min_margin: float,
) -> tuple[bool, float, float, RejectionReason | None]:
    if probs.size == 0:
        return True, 0.0, 0.0, "low_confidence"

    order = np.argsort(probs)[::-1]
    top_p = float(probs[order[0]])
    second_p = float(probs[order[1]]) if probs.size > 1 else 0.0
    margin = top_p - second_p

    if top_p < min_confidence:
        return True, top_p, margin, "low_confidence"
    if margin < min_margin:
        return True, top_p, margin, "low_margin"
    return False, top_p, margin, None


def build_alternatives(
    probs: np.ndarray,
    trainable_ids: list[str],
    disease_names: dict[str, str],
    *,
    top_k: int = 3,
) -> list[ClassScore]:
    probs = np.atleast_1d(np.asarray(probs, dtype=np.float64)).reshape(-1)
    if probs.size == 0 or not trainable_ids:
        return []
    k = min(top_k, len(trainable_ids), int(probs.size))
    order = np.argsort(probs)[::-1][:k]
    out: list[ClassScore] = []
    for i in order:
        if i >= len(trainable_ids):
            break
        cid = trainable_ids[i]
        out.append(
            ClassScore(
                class_id=cid,
                disease_name=disease_names.get(cid, cid),
                confidence_pct=round(float(probs[i]) * 100.0, 1),
            )
        )
    return out


def average_tta_logits(
    run_logits,
    image: Image.Image,
    settings: Settings,
    layout: str,
) -> np.ndarray:
    """Average raw logits across TTA views (better generalization than single crop)."""
    logits_sum: np.ndarray | None = None
    n = 0
    for variant in tta_image_variants(image):
        x = preprocess_imagenet(variant, settings.input_size, layout=layout)
        logits = normalize_logits(run_logits(x.astype(np.float32)))
        logits_sum = logits.astype(np.float64) if logits_sum is None else logits_sum + logits.astype(np.float64)
        n += 1
    assert logits_sum is not None and n > 0
    return (logits_sum / n).astype(np.float32)
