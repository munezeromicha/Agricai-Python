from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Agric AI Vision API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS: comma-separated origins, e.g. http://localhost:5173,http://127.0.0.1:5173
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # stub | onnx | keras | roboflow
    inference_mode: str = "stub"
    model_path: str | None = None
    model_version: str = "0.0.0-stub"

    # Optional override for knowledge base (tomato-only testing: data/classes_tomato.json)
    classes_path: str | None = None

    # keras only: builtin_rescale = model has Rescaling(1/255); imagenet = manual ImageNet norm
    keras_preprocess: str = "builtin_rescale"

    # ImageNet-style preprocessing (change if your training recipe differs)
    input_size: int = 224

    # Reject when top softmax score is below this (0–1). ~0.55–0.60 balances field photos vs junk.
    confidence_threshold: float = 0.58

    # Top class must beat runner-up by at least this margin (0–1), or return unknown.
    confidence_margin: float = 0.10

    # Average logits over flipped/cropped views — improves phone-photo generalization.
    tta_enabled: bool = True

    # Run leaf-plausibility checks before/after inference — blocks people, landscapes, etc.
    plant_guard_enabled: bool = True

    # Minimum 0–1 leaf score required before running the classifier (0.42 balances field photos).
    plant_guard_min_score: float = 0.42

    # After inference: reject very confident predictions when the image still looks non-leaf.
    ood_leaf_score_max: float = 0.50
    ood_confidence_trigger: float = 0.85

    # Crop to the largest leaf blob before guard + classifier (multi-leaf / busy backgrounds)
    leaf_auto_crop_enabled: bool = True

    # Stage 1 binary gate — tomato leaf vs everything else (model/tomato_leaf_gate.keras)
    tomato_gate_enabled: bool = True
    tomato_gate_path: str | None = "model/tomato_leaf_gate.keras"
    # Pass gate when tomato_leaf score >= this (0–1).
    tomato_gate_threshold: float = 0.48
    # Below this score on both full + cropped views → hard reject (clearly not tomato).
    tomato_gate_hard_reject: float = 0.22
    # Between hard_reject and threshold: still run disease model (field/garden photos).
    tomato_gate_soft_pass: bool = True
    # Use max(gate score on full photo, gate score on auto-cropped leaf).
    tomato_gate_use_cropped: bool = True
    # When plant-guard leaf score is this high, run disease model even if gate score is low.
    tomato_gate_bypass_min_leaf_score: float = 0.62

    # Stage 0 — multi-crop identity gate (local ONNX classifier, runs before the Roboflow call).
    # Covers 8/11 crops: tomato, beans, maize, mango, onion, orange, potato, tea.
    crop_gate_enabled: bool = True
    crop_gate_path: str | None = "model/archive/crop_classifier.onnx"
    crop_gate_labels_path: str | None = "model/class_names.json"
    # Crop-identity gate thresholds. These now apply to the *aggregated per-crop*
    # probability (all of a crop's classes summed), not a single diluted class score.
    # When the SELECTED crop is one the gate was trained on (tomato/beans/maize/mango/
    # onion/orange/potato/tea): reject only if some OTHER crop's aggregate score clears
    # this bar AND beats the selected crop's own score by crop_gate_margin.
    crop_gate_mismatch_threshold: float = 0.60
    crop_gate_margin: float = 0.25
    # When the SELECTED crop is one the gate can't see (coffee/cassava/banana): the
    # classifier has no class for it and was measured confidently mislabeling genuine
    # cassava leaves as "beans" at 0.985 — indistinguishable from a real maize leaf at
    # 1.000. So gate-based rejection is OFF for these crops by default (it would false-
    # reject real farmers' photos). They fall through to the Roboflow + keyword check.
    # Set True only if you accept that risk; then UNCOVERED_THRESHOLD must be very high.
    crop_gate_reject_uncovered: bool = False
    crop_gate_uncovered_threshold: float = 0.99

    # 11-class model: reject when Not_Tomato probability is competitive with top disease
    not_tomato_compete_threshold: float = 0.28
    not_tomato_compete_margin: float = 0.18

    # Roboflow serverless object detection (INFERENCE_MODE=roboflow)
    roboflow_api_url: str = "https://serverless.roboflow.com"
    roboflow_api_key: str | None = None
    roboflow_model_id: str = "tomato-disease-b518h/3"
    # Filter predictions below this confidence (0–1), matches Roboflow UI default 50%
    roboflow_confidence_threshold: float = 0.50
    roboflow_api_confidence_pct: int = 10
    roboflow_iou_threshold: float = 0.50
    roboflow_workspace_name: str = "pro-grammer"

    # Reject Roboflow-path uploads that fail local blur/resolution/brightness checks
    # before spending an API call (needs real-world tuning once you have field photos).
    image_quality_enabled: bool = True
    image_quality_blur_variance_threshold: float = 60.0
    image_quality_min_edge_px: int = 200
    image_quality_min_mean_luma: float = 25.0
    image_quality_max_mean_luma: float = 235.0

    # Roboflow-path margin check (top-1 vs top-2 class), analogous to CONFIDENCE_MARGIN
    # above but kept separate since Roboflow's confidence distribution differs from the
    # local model's softmax.
    roboflow_margin_threshold: float = 0.10

    # Adaptive test-time augmentation: only fires a second (horizontally-flipped)
    # Roboflow call when the first call's result is borderline, to bound added
    # API cost/latency to genuinely uncertain cases.
    roboflow_tta_enabled: bool = True
    # Trigger band (percentage points) around roboflow_confidence_threshold*100.
    roboflow_tta_band_pct: float = 15.0
    # Also trigger when the top-1/top-2 margin is at or below this many points, even
    # at high confidence (wider than roboflow_margin_threshold so genuinely-close
    # calls get a second look before they'd otherwise reach the margin-reject check).
    roboflow_tta_margin_trigger_pct: float = 20.0

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def resolved_classes_path(self) -> Path:
        if self.classes_path:
            p = Path(self.classes_path)
            if not p.is_absolute():
                p = self.project_root / p
            return p
        return self.project_root / "data" / "classes.json"

    def resolved_model_path(self) -> Path | None:
        if not self.model_path:
            return None
        p = Path(self.model_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def resolved_gate_path(self) -> Path | None:
        if not self.tomato_gate_path:
            return None
        p = Path(self.tomato_gate_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def resolved_crop_gate_path(self) -> Path | None:
        if not self.crop_gate_path:
            return None
        p = Path(self.crop_gate_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def resolved_crop_gate_labels_path(self) -> Path | None:
        if not self.crop_gate_labels_path:
            return None
        p = Path(self.crop_gate_labels_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p


def get_settings() -> Settings:
    """Read settings on each call so .env changes apply without a full process restart."""
    return Settings()
