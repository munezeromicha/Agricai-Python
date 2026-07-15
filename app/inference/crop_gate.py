"""Stage 0 — multi-crop identity gate: does this photo look like the selected crop at all?

Runs a local ONNX classifier (`model/archive/crop_classifier.onnx`) before the Roboflow
call so an obvious cross-crop photo (e.g. an onion leaf submitted for the tomato model)
can be rejected without spending a Roboflow API call. Covers 8 of 11 crops (tomato, beans,
maize, mango, onion, orange, potato, tea) — coffee/cassava/banana are not represented in
this classifier's training set and always come back inconclusive (`None`).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.inference.classify import infer_onnx_layout, logits_to_probs, normalize_logits, preprocess_imagenet

# Prefix match: label.lower() startswith key -> crop id.
_PREFIX_TO_CROP: dict[str, str] = {
    "tomato___": "tomato",
    "beans_": "beans",
    "maize_": "maize",
    "mango_": "mango",
    "onion_": "onion",
    "orange_": "orange",
    "potato___": "potato",
}

# Exact match (normalized): label -> crop id. These are unprefixed in class_names.json.
_EXACT_LABEL_TO_CROP: dict[str, str] = {
    "algal leaf": "tea",
    "bird eye spot": "tea",
    "brown blight": "tea",
    "gray light": "tea",
    "red leaf spot": "tea",
    "white spot": "tea",
}

# Bare labels that carry no reliable crop signal on their own — never treated as a match
# or a mismatch.
_INCONCLUSIVE_LABELS: frozenset[str] = frozenset({"healthy", "anthracnose", "bulb rot"})

# The 8 crops this classifier was actually trained on. The other 3 selectable crops
# (coffee, cassava, banana) have NO class here — the gate can never confirm their
# identity, only detect when a photo is clearly one of these 8 instead.
COVERED_CROPS: frozenset[str] = frozenset(
    {"tomato", "beans", "maize", "mango", "onion", "orange", "potato", "tea"}
)


def _label_to_crop_id(label: str) -> str | None:
    norm = label.strip().lower()
    if norm in _INCONCLUSIVE_LABELS:
        return None
    if norm in _EXACT_LABEL_TO_CROP:
        return _EXACT_LABEL_TO_CROP[norm]
    for prefix, crop_id in _PREFIX_TO_CROP.items():
        if norm.startswith(prefix):
            return crop_id
    return None


class CropIdentityGate:
    """Local ONNX classifier used only to confirm/deny the selected crop's identity."""

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_layout = infer_onnx_layout(tuple(inp.shape))
        self._output_name = self._session.get_outputs()[0].name

        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self._class_names: list[str] = list(labels["class_names"])

    def identify(
        self, image: Image.Image, settings: Settings
    ) -> tuple[str | None, float, dict[str, float]]:
        """Return (top_crop_id or None, its aggregated prob 0-1, per-crop prob map).

        Probabilities are aggregated PER CROP, not per class. A crop like maize has 4
        separate classes here (Blight/Rust/Gray Leaf Spot/Healthy); a real maize leaf can
        spread its probability across all four, so the single top-1 class prob badly
        understates how confident the model is that the crop is maize. Summing per crop
        fixes that dilution.
        """
        x = preprocess_imagenet(image, settings.input_size, layout=self._input_layout)
        raw = self._session.run([self._output_name], {self._input_name: x.astype("float32")})[0]
        probs = logits_to_probs(normalize_logits(raw))

        crop_probs: dict[str, float] = {}
        for i, label in enumerate(self._class_names):
            if i >= probs.shape[0]:
                break
            cid = _label_to_crop_id(label)
            if cid is None:  # inconclusive/unmapped class contributes to no crop
                continue
            crop_probs[cid] = crop_probs.get(cid, 0.0) + float(probs[i])

        if not crop_probs:
            return None, 0.0, {}
        top_crop = max(crop_probs, key=lambda c: crop_probs[c])
        return top_crop, crop_probs[top_crop], crop_probs


_gate_cache_key: str | None = None
_gate_cache: CropIdentityGate | None = None


def get_crop_gate(settings: Settings) -> CropIdentityGate | None:
    """Load once per (model path, labels path); returns None when disabled or files missing."""
    global _gate_cache_key, _gate_cache
    if not settings.crop_gate_enabled:
        return None
    model_path = settings.resolved_crop_gate_path()
    labels_path = settings.resolved_crop_gate_labels_path()
    if model_path is None or not model_path.is_file():
        return None
    if labels_path is None or not labels_path.is_file():
        return None

    key = f"{model_path.resolve()}|{labels_path.resolve()}"
    if _gate_cache_key != key or _gate_cache is None:
        try:
            _gate_cache = CropIdentityGate(model_path, labels_path)
            _gate_cache_key = key
        except Exception:
            _gate_cache = None
            _gate_cache_key = None
    return _gate_cache
