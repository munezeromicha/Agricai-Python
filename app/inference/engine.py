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
    is_overconfident_nonleaf,
    logits_to_probs,
    normalize_logits,
    prediction_is_uncertain,
    preprocess_imagenet,
)
from app.inference.knowledge import KnowledgeBase
from app.inference.leaf_crop import extract_primary_leaf
from app.inference.plant_guard import (
    is_obvious_scene_photo,
    leaf_plausibility_score,
    looks_like_crop_leaf_photo,
)
from app.inference.reject_messages import unknown_copy
from app.inference.tomato_gate import NOT_TOMATO_CLASS_ID, TomatoLeafGate, load_tomato_gate
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


def _plant_guard_reject() -> ClassifyDetails:
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


def _not_tomato_reject(top_confidence: float = 0.0) -> ClassifyDetails:
    return ClassifyDetails(
        probs=np.array([]),
        top_class_id=None,
        uncertain=True,
        rejection_reason="not_tomato",
        top_confidence=top_confidence,
        margin=0.0,
        alternatives=[],
        plant_guard_blocked=False,
        tomato_gate_score=top_confidence,
    )


def _tomato_gate_best_score(
    original: Image.Image,
    cropped: Image.Image,
    gate: TomatoLeafGate,
    settings: Settings,
) -> float:
    """Score 0–1; uses the better of full-frame vs auto-cropped leaf."""
    s_full = gate.tomato_leaf_score(original, settings)
    if not settings.tomato_gate_use_cropped:
        return s_full
    s_crop = gate.tomato_leaf_score(cropped, settings)
    return max(s_full, s_crop)


def _evaluate_tomato_gate(
    score: float,
    settings: Settings,
    *,
    leaf_score: float | None = None,
) -> tuple[ClassifyDetails | None, bool]:
    """
    Returns (reject details or None, soft_pass).
    soft_pass=True → run disease model even though gate score < threshold.
    """
    if score >= settings.tomato_gate_threshold:
        return None, False
    if (
        leaf_score is not None
        and leaf_score >= settings.tomato_gate_bypass_min_leaf_score
    ):
        return None, True
    if score < settings.tomato_gate_hard_reject:
        return _not_tomato_reject(top_confidence=score), False
    if settings.tomato_gate_soft_pass:
        return None, True
    return _not_tomato_reject(top_confidence=score), False


def _run_pre_model_guards(
    original: Image.Image,
    settings: Settings,
    gate: TomatoLeafGate | None,
) -> tuple[Image.Image, ClassifyDetails | None, bool, float]:
    """
  Prepare image for the disease model.

  Order: obvious-scene check → auto-crop → plant guard → tomato gate (best of full/crop).
  Field photos often fail gate on the full frame but pass on the cropped leaf.
  """
    original_rgb = original.convert("RGB")

    if settings.plant_guard_enabled and is_obvious_scene_photo(original_rgb):
        return original_rgb, _plant_guard_reject(), False, 0.0

    cropped = original_rgb
    if settings.leaf_auto_crop_enabled:
        cropped, _ = extract_primary_leaf(original_rgb)

    blocked = _check_plant_guard(cropped, settings)
    if blocked is not None:
        return cropped, blocked, False, 0.0

    leaf_score = leaf_plausibility_score(cropped)

    if gate is not None:
        score = _tomato_gate_best_score(original_rgb, cropped, gate, settings)
        blocked, soft = _evaluate_tomato_gate(score, settings, leaf_score=leaf_score)
        if blocked is not None:
            return cropped, blocked, False, score
        return cropped, None, soft, score

    return cropped, None, False, 1.0


def _prepare_image_for_inference(
    image: Image.Image, settings: Settings
) -> tuple[Image.Image, ClassifyDetails | None]:
    """Legacy helper — prefer _run_pre_model_guards for full pipeline."""
    if settings.plant_guard_enabled and is_obvious_scene_photo(image):
        return image, _plant_guard_reject()
    if settings.leaf_auto_crop_enabled:
        cropped, _ = extract_primary_leaf(image)
        return cropped, None
    return image, None


def _check_plant_guard(image: Image.Image, settings: Settings) -> ClassifyDetails | None:
    if not settings.plant_guard_enabled:
        return None
    if not looks_like_crop_leaf_photo(image, min_score=settings.plant_guard_min_score):
        return _plant_guard_reject()
    return None


def _finalize_probs(
    image: Image.Image,
    settings: Settings,
    probs: np.ndarray,
    trainable_ids: list[str],
    disease_names: dict[str, str],
    expected_classes: int,
    *,
    tomato_gate_soft_pass: bool = False,
    tomato_gate_score: float | None = None,
) -> ClassifyDetails:
    alts = build_alternatives(probs, trainable_ids, disease_names)

    if probs.shape[0] != expected_classes:
        return ClassifyDetails(
            probs=probs,
            top_class_id=None,
            uncertain=True,
            rejection_reason="class_count_mismatch",
            top_confidence=float(np.max(probs)) if probs.size else 0.0,
            margin=0.0,
            alternatives=alts,
            tomato_gate_score=tomato_gate_score,
            tomato_gate_soft_pass=tomato_gate_soft_pass,
        )

    best_i = int(np.argmax(probs))
    if best_i >= len(trainable_ids):
        return ClassifyDetails(
            probs=probs,
            top_class_id=None,
            uncertain=True,
            rejection_reason="class_count_mismatch",
            top_confidence=float(probs[best_i]),
            margin=0.0,
            alternatives=alts,
            tomato_gate_score=tomato_gate_score,
            tomato_gate_soft_pass=tomato_gate_soft_pass,
        )

    # Gate soft-pass: prefer a confident disease label over Not_Tomato for field photos.
    if (
        tomato_gate_soft_pass
        and NOT_TOMATO_CLASS_ID in trainable_ids
        and trainable_ids[best_i] == NOT_TOMATO_CLASS_ID
    ):
        disease_indices = [
            i for i, cid in enumerate(trainable_ids) if cid != NOT_TOMATO_CLASS_ID
        ]
        if disease_indices:
            di = max(disease_indices, key=lambda i: float(probs[i]))
            if float(probs[di]) >= settings.confidence_threshold:
                best_i = di

    if trainable_ids[best_i] == NOT_TOMATO_CLASS_ID:
        disease_alts = [a for a in alts if a.class_id != NOT_TOMATO_CLASS_ID]
        disease_indices = [
            i for i, cid in enumerate(trainable_ids) if cid != NOT_TOMATO_CLASS_ID
        ]
        leaf_score = leaf_plausibility_score(image)
        field_tomato_leaf = (
            tomato_gate_soft_pass
            and settings.plant_guard_enabled
            and leaf_score >= settings.tomato_gate_bypass_min_leaf_score
            and disease_indices
        )
        if field_tomato_leaf:
            di = max(disease_indices, key=lambda i: float(probs[i]))
            best_p = float(probs[di])
            ranked = sorted((float(probs[i]) for i in disease_indices), reverse=True)
            margin = best_p - (ranked[1] if len(ranked) > 1 else 0.0)
            return ClassifyDetails(
                probs=probs,
                top_class_id=None,
                uncertain=True,
                rejection_reason="low_confidence",
                top_confidence=best_p,
                margin=margin,
                alternatives=disease_alts,
                tomato_gate_score=tomato_gate_score,
                tomato_gate_soft_pass=tomato_gate_soft_pass,
            )
        return ClassifyDetails(
            probs=probs,
            top_class_id=None,
            uncertain=True,
            rejection_reason="not_tomato",
            top_confidence=float(probs[best_i]),
            margin=0.0,
            alternatives=disease_alts,
            tomato_gate_score=tomato_gate_score,
            tomato_gate_soft_pass=tomato_gate_soft_pass,
        )

    if NOT_TOMATO_CLASS_ID in trainable_ids:
        nt_i = trainable_ids.index(NOT_TOMATO_CLASS_ID)
        nt_p = float(probs[nt_i])
        top_p = float(probs[best_i])
        compete_thresh = settings.not_tomato_compete_threshold
        compete_margin = settings.not_tomato_compete_margin
        if tomato_gate_soft_pass:
            compete_thresh = min(0.55, compete_thresh + 0.12)
            compete_margin = min(0.28, compete_margin + 0.06)
        if nt_p >= compete_thresh and (top_p - nt_p) < compete_margin:
            disease_alts = [a for a in alts if a.class_id != NOT_TOMATO_CLASS_ID]
            if tomato_gate_soft_pass and top_p >= settings.confidence_threshold:
                reason: RejectionReason = "low_margin"
            else:
                return ClassifyDetails(
                    probs=probs,
                    top_class_id=None,
                    uncertain=True,
                    rejection_reason="not_tomato",
                    top_confidence=nt_p,
                    margin=top_p - nt_p,
                    alternatives=disease_alts,
                    tomato_gate_score=tomato_gate_score,
                    tomato_gate_soft_pass=tomato_gate_soft_pass,
                )
            return ClassifyDetails(
                probs=probs,
                top_class_id=None,
                uncertain=True,
                rejection_reason=reason,
                top_confidence=top_p,
                margin=top_p - nt_p,
                alternatives=disease_alts,
                tomato_gate_score=tomato_gate_score,
                tomato_gate_soft_pass=tomato_gate_soft_pass,
            )

    uncertain, top_p, margin, reason = prediction_is_uncertain(
        probs,
        min_confidence=settings.confidence_threshold,
        min_margin=settings.confidence_margin,
    )

    if (
        settings.plant_guard_enabled
        and not uncertain
        and is_overconfident_nonleaf(
            probs,
            leaf_plausibility_score(image),
            min_leaf_score=settings.ood_leaf_score_max,
            confidence_trigger=settings.ood_confidence_trigger,
        )
    ):
        return _plant_guard_reject()

    class_id = None if uncertain else trainable_ids[best_i]
    disease_alts = [a for a in alts if a.class_id != NOT_TOMATO_CLASS_ID]
    return ClassifyDetails(
        probs=probs,
        top_class_id=class_id,
        uncertain=uncertain,
        rejection_reason=reason,
        top_confidence=top_p,
        margin=margin,
        alternatives=disease_alts,
        tomato_gate_score=tomato_gate_score,
        tomato_gate_soft_pass=tomato_gate_soft_pass,
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

    def __init__(
        self, kb: KnowledgeBase, settings: Settings, gate: TomatoLeafGate | None = None
    ) -> None:
        self.kb = kb
        self._settings = settings
        self._gate = gate
        self._rotating = [cid for cid in kb.disease_class_ids if cid != "unknown"]
        if not self._rotating:
            self._rotating = list(kb.class_ids)
        self._disease_names = {cid: kb.get(cid).diseaseName for cid in kb.trainable_class_ids}

    def classify(self, image: Image.Image) -> ClassifyDetails:
        settings = get_settings()
        gate = _resolve_gate(settings)
        image, blocked, gate_soft, gate_score = _run_pre_model_guards(
            image, settings, gate
        )
        if blocked is not None:
            return blocked

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

        return _finalize_probs(
            image,
            settings,
            probs,
            self._rotating,
            self._disease_names,
            len(self._rotating),
            tomato_gate_soft_pass=gate_soft,
            tomato_gate_score=gate_score,
        )


class OnnxEngine(InferenceEngine):
    """ONNX Runtime classifier; class order must match `classes.json` trainable order."""

    def __init__(
        self,
        kb: KnowledgeBase,
        settings: Settings,
        model_path: Path,
        gate: TomatoLeafGate | None = None,
    ) -> None:
        import onnxruntime as ort

        self.kb = kb
        self._settings = settings
        self._gate = gate
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
        settings = get_settings()
        gate = _resolve_gate(settings)
        image, blocked, gate_soft, gate_score = _run_pre_model_guards(
            image, settings, gate
        )
        if blocked is not None:
            return blocked

        if settings.tta_enabled:
            logits = average_tta_logits(
                self._run_logits, image, settings, self._input_layout
            )
        else:
            x = preprocess_imagenet(
                image, settings.input_size, layout=self._input_layout
            )
            logits = self._run_logits(x.astype(np.float32))

        probs = logits_to_probs(logits)
        return _finalize_probs(
            image,
            settings,
            probs,
            self._trainable_ids,
            self._disease_names,
            self._expected_classes,
            tomato_gate_soft_pass=gate_soft,
            tomato_gate_score=gate_score,
        )


_gate_cache_key: str | None = None
_gate_cache: TomatoLeafGate | None = None


def _resolve_gate(settings: Settings) -> TomatoLeafGate | None:
    """Load gate once per model path; thresholds still read from fresh settings."""
    global _gate_cache_key, _gate_cache
    if not settings.tomato_gate_enabled:
        return None
    path = settings.resolved_gate_path()
    if path is None or not path.is_file():
        return None
    key = str(path.resolve())
    if _gate_cache_key != key or _gate_cache is None:
        _gate_cache_key = key
        _gate_cache = TomatoLeafGate(settings, path)
    return _gate_cache


def get_engine(settings: Settings | None = None) -> InferenceEngine:
    settings = settings or get_settings()
    kb = KnowledgeBase(settings.resolved_classes_path)
    gate = load_tomato_gate(settings)

    if settings.tomato_gate_enabled and gate is None:
        gate_path = settings.resolved_gate_path()
        import warnings

        warnings.warn(
            f"TOMATO_GATE_ENABLED but gate model not found at {gate_path}. "
            "Non-tomato images may receive wrong disease labels.",
            stacklevel=2,
        )

    mode = settings.inference_mode.lower().strip()
    if mode == "onnx":
        if not settings.model_path:
            raise RuntimeError("INFERENCE_MODE=onnx requires MODEL_PATH to an .onnx file.")
        path = settings.resolved_model_path()
        if path is None or not path.is_file():
            raise FileNotFoundError(f"MODEL_PATH not found: {settings.model_path}")
        return OnnxEngine(kb, settings, path, gate=gate)

    if mode == "keras":
        if not settings.model_path:
            raise RuntimeError("INFERENCE_MODE=keras requires MODEL_PATH to a .keras file.")
        path = settings.resolved_model_path()
        if path is None or not path.is_file():
            raise FileNotFoundError(f"MODEL_PATH not found: {settings.model_path}")
        return KerasEngine(kb, settings, path, gate=gate)

    return StubEngine(kb, settings, gate=gate)


def _preprocess_for_keras(image: Image.Image, settings: Settings) -> np.ndarray:
    """Build batch input for .keras models."""
    image = image.convert("RGB")
    image = image.resize((settings.input_size, settings.input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    mode = settings.keras_preprocess.lower().strip()
    if mode == "imagenet":
        arr = arr / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
    # builtin_rescale: pixels 0–255; model's Rescaling(1/255) layer handles normalization
    return np.expand_dims(arr, axis=0)


class KerasEngine(InferenceEngine):
    """TensorFlow/Keras .keras classifier (tomato-only testing)."""

    def __init__(
        self,
        kb: KnowledgeBase,
        settings: Settings,
        model_path: Path,
        gate: TomatoLeafGate | None = None,
    ) -> None:
        import tensorflow as tf

        self.kb = kb
        self._settings = settings
        self._gate = gate
        self._model = tf.keras.models.load_model(model_path, compile=False)
        last = self._model.layers[-1]
        activation = getattr(last, "activation", None)
        act_name = getattr(activation, "__name__", str(activation))
        self._outputs_softmax = "softmax" in act_name.lower()
        self._trainable_ids = kb.trainable_class_ids
        self._expected_classes = kb.num_trainable_classes
        self._disease_names = {cid: kb.get(cid).diseaseName for cid in self._trainable_ids}

    def _run_probs(self, batch: np.ndarray) -> np.ndarray:
        raw = self._model.predict(batch, verbose=0)
        out = np.asarray(raw, dtype=np.float32).reshape(-1)
        if self._outputs_softmax:
            return out
        return logits_to_probs(out)

    def classify(self, image: Image.Image) -> ClassifyDetails:
        settings = get_settings()
        gate = _resolve_gate(settings)
        image, blocked, gate_soft, gate_score = _run_pre_model_guards(
            image, settings, gate
        )
        if blocked is not None:
            return blocked

        if settings.tta_enabled:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            p1 = self._run_probs(_preprocess_for_keras(image, settings))
            p2 = self._run_probs(_preprocess_for_keras(flipped, settings))
            probs = (p1 + p2) / 2.0
        else:
            probs = self._run_probs(_preprocess_for_keras(image, settings))

        return _finalize_probs(
            image,
            settings,
            probs,
            self._trainable_ids,
            self._disease_names,
            self._expected_classes,
            tomato_gate_soft_pass=gate_soft,
            tomato_gate_score=gate_score,
        )


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
        tomato_gate_score_pct=(
            round(details.tomato_gate_score * 100.0, 1)
            if details.tomato_gate_score is not None
            else None
        ),
        tomato_gate_soft_pass=details.tomato_gate_soft_pass or None,
    )
