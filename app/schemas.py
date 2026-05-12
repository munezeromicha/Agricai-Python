from typing import Literal

from pydantic import BaseModel, Field

DiseaseType = Literal["healthy", "disease", "pest", "unknown"]


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


class DetectResponse(BaseModel):
    result: DetectionResult
    model_version: str
    request_id: str
    inference_mode: str
    top_class_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    inference_mode: str


class ModelInfoResponse(BaseModel):
    model_version: str
    inference_mode: str
    input_size: int
    num_classes: int
