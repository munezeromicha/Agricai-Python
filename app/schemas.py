from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.inference.confidence import confidence_guidance, confidence_level

DiseaseType = Literal["healthy", "disease", "pest", "unknown"]
RejectionReason = Literal[
    "plant_guard",
    "not_tomato",
    "wrong_crop",
    "unsupported_crop",
    "image_quality",
    "unstable_prediction",
    "low_confidence",
    "low_margin",
    "class_count_mismatch",
]


class DetectionResult(BaseModel):
    """Matches the frontend `DetectionResult` in `src/pages/Detect.tsx`."""

    diseaseName: str
    diseaseNameRw: str
    confidence: float = Field(..., ge=0, le=100)
    type: DiseaseType
    explanation: str
    explanationRw: str
    treatment: str
    treatmentRw: str
    prevention: str
    preventionRw: str
    care: str
    careRw: str


class ClassAlternative(BaseModel):
    class_id: str
    disease_name: str
    confidence: float = Field(..., ge=0, le=100)


class DetectionBox(BaseModel):
    """Roboflow object-detection region (center x/y and size in image pixels)."""

    class_name: str
    class_id: str | None = None
    confidence: float = Field(..., ge=0, le=100)
    x: float
    y: float
    width: float
    height: float
    color: str = "#14b8a6"


class DetectResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    result: DetectionResult
    model_version: str
    request_id: str
    inference_mode: str
    top_class_id: str | None = None
    rejection_reason: RejectionReason | None = None
    alternatives: list[ClassAlternative] = Field(default_factory=list)
    top_confidence_pct: float | None = None
    confidence_margin_pct: float | None = None
    tomato_gate_score_pct: float | None = None
    tomato_gate_soft_pass: bool | None = None
    tta_ran: bool | None = None
    tta_agreed: bool | None = None
    detections: list[DetectionBox] = Field(default_factory=list)
    image_width: int | None = None
    image_height: int | None = None
    roboflow_model_id: str | None = None
    crop_id: str | None = None
    #: Band derived from `top_confidence_pct` + `confidence_margin_pct` — the app shows
    #: this word rather than making each client re-derive its own thresholds.
    confidence_level: Literal["high", "medium", "low", "very_low"] = "very_low"
    confidence_guidance: str = ""
    confidence_guidance_rw: str = ""
    actionable: bool = False

    @model_validator(mode="after")
    def _derive_confidence(self) -> "DetectResponse":
        # A rejected result carries no usable diagnosis, so it never reports a band above very_low.
        if self.rejection_reason is not None or self.result.type == "unknown":
            level = "very_low"
        else:
            score = self.top_confidence_pct if self.top_confidence_pct is not None else self.result.confidence
            level = confidence_level(score, self.confidence_margin_pct)
        en, rw = confidence_guidance(level)
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "confidence_guidance", en)
        object.__setattr__(self, "confidence_guidance_rw", rw)
        object.__setattr__(self, "actionable", level in ("high", "medium"))
        return self


class CropSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    display_name: str
    inference_kind: str
    model_id: str | None = None
    workflow_configured: bool = False


class CropsListResponse(BaseModel):
    crops: list[CropSummary]
    default_crop_id: str = "tomato"


class ValidateLeafResponse(BaseModel):
    crop_id: str
    status: Literal["match", "mismatch", "uncertain", "no_detection"]
    crop_match: bool
    top_class: str | None = None
    top_confidence_pct: float | None = None
    message: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    inference_mode: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    inference_mode: str
    input_size: int
    num_classes: int
    plant_guard_enabled: bool = True
    tta_enabled: bool = True
    confidence_threshold: float = 0.58
    confidence_margin: float = 0.10
    val_accuracy_pct: float | None = None
    target_accuracy_pct: float = 98.0


class ClassSummary(BaseModel):
    class_id: str
    type: DiseaseType
    diseaseName: str
    diseaseNameRw: str


class ClassesListResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    count: int
    classes: list[ClassSummary]


class ClassDetailResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    class_id: str
    result: DetectionResult
