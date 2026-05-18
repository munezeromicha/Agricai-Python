from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.inference.knowledge import KnowledgeBase
from app.inference.plant_guard import looks_like_crop_leaf_photo
from app.schemas import DetectResponse, DetectionResult


def _channel_dim_index(shape: tuple) -> int | None:
    """Return axis index (1..3) where channel size is 3, for 4D ONNX inputs."""
    if len(shape) != 4:
        return None
    for i in (1, 2, 3):
        dim = shape[i]
        if dim == 3 or dim == "3":
            return i
    return None


def _infer_onnx_layout(input_shape: tuple) -> str:
    """Keras/tf2onnx exports use NHWC [N,H,W,3]; some ONNX models use NCHW [N,3,H,W]."""
    ch = _channel_dim_index(input_shape)
    if ch == 1:
        return "nchw"
    if ch == 3:
        return "nhwc"
    # Dynamic shapes from Keras often look like (batch, 224, 224, 3)
    return "nhwc"


def _preprocess_imagenet(image: Image.Image, size: int, layout: str = "nhwc") -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std  # HWC
    if layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)


def _prediction_is_uncertain(
    probs: np.ndarray,
    *,
    min_confidence: float,
    min_margin: float,
) -> tuple[bool, float, float]:
    """
  Decide if we should show a specific disease vs the generic "unknown" entry.

  Rules (all must pass to accept a label):
    1. Top score >= min_confidence (default 65%) — blocks weak guesses on non-leaf images.
    2. Top score - second score >= min_margin (default 12%) — blocks "almost tied" guesses.
    """
    if probs.size == 0:
        return True, 0.0, 0.0

    order = np.argsort(probs)[::-1]
    top_p = float(probs[order[0]])
    second_p = float(probs[order[1]]) if probs.size > 1 else 0.0
    margin = top_p - second_p

    if top_p < min_confidence:
        return True, top_p, margin
    if margin < min_margin:
        return True, top_p, margin
    return False, top_p, margin


def _reject_unknown(kb: KnowledgeBase, confidence_pct: float = 0.0) -> tuple[DetectionResult, str]:
    unk = kb.get("unknown")
    return kb.to_detection(unk, confidence_pct), "unknown"


def _plant_guard_reject(settings: Settings, kb: KnowledgeBase, image: Image.Image) -> tuple[DetectionResult, str] | None:
    if not settings.plant_guard_enabled:
        return None
    if looks_like_crop_leaf_photo(image):
        return None
    return _reject_unknown(kb, 0.0)


class InferenceEngine(ABC):
    kb: KnowledgeBase

    @abstractmethod
    def predict(self, image: Image.Image) -> tuple[DetectionResult, str | None]:
        """Return detection result and top_class_id (knowledge key)."""


class StubEngine(InferenceEngine):
    """Deterministic demo inference without a model file.

    Picks a class index from a hash of raw bytes so the same upload gives the same label.
    """

    def __init__(self, kb: KnowledgeBase, settings: Settings) -> None:
        self.kb = kb
        self._settings = settings
        # Exclude generic unknown from rotation unless it's the only option
        self._rotating = [cid for cid in kb.class_ids if cid != "unknown"]
        if not self._rotating:
            self._rotating = list(kb.class_ids)

    def predict(self, image: Image.Image) -> tuple[DetectionResult, str | None]:
        blocked = _plant_guard_reject(self._settings, self.kb, image)
        if blocked is not None:
            return blocked
        buf = image.tobytes()
        h = int(hashlib.sha256(buf).hexdigest(), 16)
        idx = h % len(self._rotating) if self._rotating else 0
        class_id = self._rotating[idx]
        entry = self.kb.get(class_id)
        # Simulated confidence 72–98%
        conf = 72.0 + (h % 2700) / 100.0
        conf = min(98.0, max(72.0, conf))
        return self.kb.to_detection(entry, conf), class_id


class OnnxEngine(InferenceEngine):
    """ONNX Runtime classifier; class order must match `classes.json` order."""

    def __init__(self, kb: KnowledgeBase, settings: Settings, model_path: Path) -> None:
        import onnxruntime as ort

        self.kb = kb
        self._settings = settings
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_layout = _infer_onnx_layout(tuple(inp.shape))
        outs = self._session.get_outputs()
        self._output_name = outs[0].name
        self._trainable_ids = kb.trainable_class_ids
        self._expected_classes = kb.num_trainable_classes

    def predict(self, image: Image.Image) -> tuple[DetectionResult, str | None]:
        blocked = _plant_guard_reject(self._settings, self.kb, image)
        if blocked is not None:
            return blocked

        x = _preprocess_imagenet(
            image, self._settings.input_size, layout=self._input_layout
        )
        logits = self._session.run([self._output_name], {self._input_name: x.astype(np.float32)})[0]
        probs = _softmax(logits[0])
        if probs.shape[0] != self._expected_classes:
            return _reject_unknown(self.kb, 0.0)

        best_i = int(np.argmax(probs))
        if best_i >= len(self._trainable_ids):
            return _reject_unknown(self.kb, 0.0)

        uncertain, max_p, _margin = _prediction_is_uncertain(
            probs,
            min_confidence=self._settings.confidence_threshold,
            min_margin=self._settings.confidence_margin,
        )
        if uncertain:
            return _reject_unknown(self.kb, max_p * 100.0)

        class_id = self._trainable_ids[best_i]
        entry = self.kb.get(class_id)
        return self.kb.to_detection(entry, max_p * 100.0), class_id


def get_engine(settings: Settings | None = None) -> InferenceEngine:
    settings = settings or get_settings()
    kb = KnowledgeBase(settings.classes_path)

    mode = settings.inference_mode.lower().strip()
    if mode == "onnx":
        if not settings.model_path:
            raise RuntimeError("INFERENCE_MODE=onnx requires MODEL_PATH to an .onnx file.")
        path = settings.resolved_model_path()
        if path is None or not path.is_file():
            raise FileNotFoundError(f"MODEL_PATH not found: {settings.model_path}")
        return OnnxEngine(kb, settings, path)

    return StubEngine(kb, settings)


def run_detect_with_engine(engine: InferenceEngine, image: Image.Image, settings: Settings | None = None) -> DetectResponse:
    settings = settings or get_settings()
    result, top_id = engine.predict(image)
    return DetectResponse(
        result=result,
        model_version=settings.model_version,
        request_id=str(uuid.uuid4()),
        inference_mode=settings.inference_mode,
        top_class_id=top_id,
    )
