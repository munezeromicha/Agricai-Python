"""Stage 1 — binary gate: is this a tomato leaf photo?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings

NOT_TOMATO_CLASS_ID = "Not_Tomato"
GATE_LABELS = ["not_tomato", "tomato_leaf"]
TOMATO_LEAF_INDEX = GATE_LABELS.index("tomato_leaf")


def preprocess_gate_image(image: Image.Image, settings: Settings) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((settings.input_size, settings.input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0)


class TomatoLeafGate:
    """Binary classifier: tomato leaf vs not tomato."""

    def __init__(self, settings: Settings, model_path: Path) -> None:
        import tensorflow as tf

        self._settings = settings
        self._model = tf.keras.models.load_model(model_path, compile=False)
        last = self._model.layers[-1]
        activation = getattr(last, "activation", None)
        act_name = getattr(activation, "__name__", str(activation))
        self._outputs_softmax = "softmax" in act_name.lower()

    def _tomato_probability(self, batch: np.ndarray) -> float:
        raw = self._model.predict(batch, verbose=0)
        out = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not self._outputs_softmax:
            out = out - np.max(out)
            out = np.exp(out)
            out = out / np.sum(out)
        if out.size <= TOMATO_LEAF_INDEX:
            return 0.0
        return float(out[TOMATO_LEAF_INDEX])

    def tomato_leaf_score(self, image: Image.Image) -> float:
        """Return 0–1 probability that the image is a tomato leaf."""
        batch = preprocess_gate_image(image, self._settings)
        if self._settings.tta_enabled:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            p1 = self._tomato_probability(batch)
            p2 = self._tomato_probability(preprocess_gate_image(flipped, self._settings))
            return (p1 + p2) / 2.0
        return self._tomato_probability(batch)

    def is_tomato_leaf(self, image: Image.Image) -> tuple[bool, float]:
        score = self.tomato_leaf_score(image)
        return score >= self._settings.tomato_gate_threshold, score


def load_tomato_gate(settings: Settings) -> TomatoLeafGate | None:
    if not settings.tomato_gate_enabled:
        return None
    path = settings.resolved_gate_path()
    if path is None or not path.is_file():
        return None
    return TomatoLeafGate(settings, path)
