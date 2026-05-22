from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.inference.classify import (
    ClassifyDetails,
    RejectionReason,
    average_tta_logits,
    build_alternatives,
    infer_onnx_layout,
    logits_to_probs,
    normalize_logits,
    prediction_is_uncertain,
    preprocess_imagenet,
)
from app.inference.knowledge import KnowledgeBase
from app.inference.plant_guard import looks_like_crop_leaf_photo
from app.inference.reject_messages import unknown_copy
from app.schemas import ClassAlternative, DetectResponse, DetectionResult


def _reject_unknown(
    kb: KnowledgeBase,
    confidence_pct: float = 0.0,
    *,
    reason: RejectionReason | None = None,
) -> tuple[DetectionResult, str]:
    name_en, name_rw, expl_en, expl_rw = unknown_copy(reason)
    unk = kb.get("unknown")
    return (
        DetectionResult(
            diseaseName=name_en,
            diseaseNameRw=name_rw,
            confidence=round(confidence_pct, 1),
            type="unknown",
            explanation=expl_en,
            explanationRw=expl_rw,
            treatment=unk.treatment,
            treatmentRw=unk.treatmentRw,
            prevention=unk.prevention,
            preventionRw=unk.preventionRw,
            care=unk.care,
            careRw=unk.careRw,
        ),
        "unknown",
    )


def _alternatives_to_schema(alternatives: list) -> list[ClassAlternative]:
    return [
        ClassAlternative(
            class_id=a.class_id,
            disease_name=a.disease_name,
            confidence=a.confidence_pct,
        )
        for a in alternatives
    ]


class InferenceEngine(ABC):
    kb: KnowledgeBase

    @abstractmethod
    def classify(self, image: Image.Image) -> ClassifyDetails:
        """Return probabilities and gate metadata (no knowledge-base copy yet)."""

    def predict(self, image: Image.Image) -> tuple[DetectionResult, str | None]:
        return self._details_to_detection(self.classify(image))

    def _details_to_detection(
        self, details: ClassifyDetails
    ) -> tuple[DetectionResult, str | None]:
        kb = self.kb
        if details.plant_guard_blocked:
            return _reject_unknown(kb, 0.0, reason="plant_guard")
        if details.uncertain or details.top_class_id is None:
            return _reject_unknown(
                kb,
                details.top_confidence * 100.0,
                reason=details.rejection_reason,
            )
        entry = kb.get(details.top_class_id)
        return kb.to_detection(entry, details.top_confidence * 100.0), details.top_class_id


class StubEngine(InferenceEngine):
    """Deterministic demo inference without a model file."""

    def __init__(self, kb: KnowledgeBase, settings: Settings) -> None:
        self.kb = kb
        self._settings = settings
        self._rotating = [cid for cid in kb.class_ids if cid != "unknown"]
        if not self._rotating:
            self._rotating = list(kb.class_ids)
        self._disease_names = {cid: kb.get(cid).diseaseName for cid in kb.trainable_class_ids}

    def classify(self, image: Image.Image) -> ClassifyDetails:
        if self._settings.plant_guard_enabled and not looks_like_crop_leaf_photo(image):
            return ClassifyDetails(
                probs=np.array([]),
                top_class_id=None,
                uncertain=True,
                rejection_reason="plant_guard",
                top_confidence=0.0,
                margin=0.0,
                alternatives=[],
                plant_guard_blocked=True,
            )

        buf = image.tobytes()
        h = int(hashlib.sha256(buf).hexdigest(), 16)
        n = len(self._rotating) or 1
        probs = np.zeros(n, dtype=np.float32)
        idx = h % n
        probs[idx] = 0.85
        if n > 1:
            probs[(idx + 1) % n] = 0.10
        for i in range(n):
            if probs[i] == 0:
                probs[i] = 0.05 / max(1, n - 2)

        uncertain, top_p, margin, reason = prediction_is_uncertain(
            probs,
            min_confidence=self._settings.confidence_threshold,
            min_margin=self._settings.confidence_margin,
        )
        top_id = self._rotating[int(np.argmax(probs))] if not uncertain else None
        alts = build_alternatives(probs, self._rotating, self._disease_names)
        return ClassifyDetails(
            probs=probs,
            top_class_id=top_id,
            uncertain=uncertain,
            rejection_reason=reason,
            top_confidence=top_p,
            margin=margin,
            alternatives=alts,
        )


class OnnxEngine(InferenceEngine):
    """ONNX Runtime classifier; class order must match `classes.json` trainable order."""

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
        self._input_layout = infer_onnx_layout(tuple(inp.shape))
        outs = self._session.get_outputs()
        self._output_name = outs[0].name
        self._trainable_ids = kb.trainable_class_ids
        self._expected_classes = kb.num_trainable_classes
        self._disease_names = {cid: kb.get(cid).diseaseName for cid in self._trainable_ids}

    def _run_logits(self, batch: np.ndarray) -> np.ndarray:
        out = self._session.run([self._output_name], {self._input_name: batch})[0]
        return normalize_logits(out)

    def classify(self, image: Image.Image) -> ClassifyDetails:
        if self._settings.plant_guard_enabled and not looks_like_crop_leaf_photo(image):
            return ClassifyDetails(
                probs=np.array([]),
                top_class_id=None,
                uncertain=True,
                rejection_reason="plant_guard",
                top_confidence=0.0,
                margin=0.0,
                alternatives=[],
                plant_guard_blocked=True,
            )

        if self._settings.tta_enabled:
            logits = average_tta_logits(
                self._run_logits, image, self._settings, self._input_layout
            )
        else:
            x = preprocess_imagenet(
                image, self._settings.input_size, layout=self._input_layout
            )
            logits = self._run_logits(x.astype(np.float32))

        probs = logits_to_probs(logits)
        alts = build_alternatives(probs, self._trainable_ids, self._disease_names)

        if probs.shape[0] != self._expected_classes:
            return ClassifyDetails(
                probs=probs,
                top_class_id=None,
                uncertain=True,
                rejection_reason="class_count_mismatch",
                top_confidence=float(np.max(probs)) if probs.size else 0.0,
                margin=0.0,
                alternatives=alts,
            )

        best_i = int(np.argmax(probs))
        if best_i >= len(self._trainable_ids):
            return ClassifyDetails(
                probs=probs,
                top_class_id=None,
                uncertain=True,
                rejection_reason="class_count_mismatch",
                top_confidence=float(probs[best_i]),
                margin=0.0,
                alternatives=alts,
            )

        uncertain, top_p, margin, reason = prediction_is_uncertain(
            probs,
            min_confidence=self._settings.confidence_threshold,
            min_margin=self._settings.confidence_margin,
        )
        class_id = None if uncertain else self._trainable_ids[best_i]
        return ClassifyDetails(
            probs=probs,
            top_class_id=class_id,
            uncertain=uncertain,
            rejection_reason=reason,
            top_confidence=top_p,
            margin=margin,
            alternatives=alts,
        )


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


def run_detect_with_engine(
    engine: InferenceEngine, image: Image.Image, settings: Settings | None = None
) -> DetectResponse:
    settings = settings or get_settings()
    details = engine.classify(image)
    result, top_id = engine._details_to_detection(details)
    return DetectResponse(
        result=result,
        model_version=settings.model_version,
        request_id=str(uuid.uuid4()),
        inference_mode=settings.inference_mode,
        top_class_id=top_id,
        rejection_reason=details.rejection_reason if details.uncertain else None,
        alternatives=_alternatives_to_schema(details.alternatives),
        top_confidence_pct=round(details.top_confidence * 100.0, 1),
        confidence_margin_pct=round(details.margin * 100.0, 1),
    )
