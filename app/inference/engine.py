from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.inference.knowledge import KnowledgeBase
from app.schemas import DetectResponse, DetectionResult


def _preprocess_imagenet(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)


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
        outs = self._session.get_outputs()
        self._output_name = outs[0].name
        self._expected_classes = kb.num_classes

    def predict(self, image: Image.Image) -> tuple[DetectionResult, str | None]:
        x = _preprocess_imagenet(image, self._settings.input_size)
        logits = self._session.run([self._output_name], {self._input_name: x.astype(np.float32)})[0]
        probs = _softmax(logits[0])
        if probs.shape[0] != self._expected_classes:
            unk = self.kb.get("unknown")
            return self.kb.to_detection(unk, 0.0), "unknown"

        best_i = int(np.argmax(probs))
        max_p = float(probs[best_i])
        class_ids = self.kb.class_ids
        if best_i >= len(class_ids):
            unk = self.kb.get("unknown")
            return self.kb.to_detection(unk, 0.0), "unknown"

        class_id = class_ids[best_i]
        if max_p < self._settings.confidence_threshold:
            unk = self.kb.get("unknown")
            # Keep raw max probability as hint in UI if you later expose it
            return self.kb.to_detection(unk, max_p * 100.0), "unknown"

        entry = self.kb.get(class_id)
        return self.kb.to_detection(entry, max_p * 100.0), class_id


def get_engine(settings: Settings | None = None) -> InferenceEngine:
    settings = settings or get_settings()
    kb = KnowledgeBase(settings.classes_path)

    mode = settings.inference_mode.lower().strip()
    if mode == "onnx":
        if not settings.model_path:
            raise RuntimeError("INFERENCE_MODE=onnx requires MODEL_PATH to an .onnx file.")
        path = Path(settings.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MODEL_PATH not found: {path}")
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
