from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DiseaseType = Literal["healthy", "disease", "pest", "unknown"]
RejectionReason = Literal[
    "plant_guard",
    "not_tomato",
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
