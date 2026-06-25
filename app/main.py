import json
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from PIL import ImageOps, UnidentifiedImageError

from app.config import get_settings
from app.inference.engine import InferenceEngine, get_engine, run_detect_with_engine
from app.inference.knowledge import KnowledgeBase
from app.inference.roboflow import run_roboflow_detect
from app.schemas import (
    ClassDetailResponse,
    ClassesListResponse,
    ClassSummary,
    DetectResponse,
    HealthResponse,
    ModelInfoResponse,
)

_engine: InferenceEngine | None = None


def _load_val_accuracy_pct() -> float | None:
    model_dir = Path(__file__).resolve().parent.parent / "model"
    for name in ("tomato_training_summary.json", "training_summary.json"):
        summary_path = model_dir / name
        if not summary_path.is_file():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            acc = data.get("metrics", {}).get("val_accuracy")
            if acc is not None:
                return round(float(acc) * 100, 1)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    settings = get_settings()
    if settings.inference_mode.lower().strip() == "roboflow":
        _engine = None
    else:
        _engine = get_engine()
    yield
    _engine = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "message": settings.app_name,
            "status": "ok",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        cfg = get_settings()
        return HealthResponse(
            status="ok",
            model_version=cfg.model_version,
            inference_mode=cfg.inference_mode,
        )

    @app.get("/v1/models", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        settings = get_settings()
        if settings.inference_mode.lower().strip() == "roboflow":
            kb = KnowledgeBase(settings.resolved_classes_path)
            return ModelInfoResponse(
                model_version=settings.model_version,
                inference_mode="roboflow",
                input_size=settings.input_size,
                num_classes=kb.num_trainable_classes,
                plant_guard_enabled=False,
                tta_enabled=False,
                confidence_threshold=settings.roboflow_confidence_threshold,
                confidence_margin=0.0,
                val_accuracy_pct=None,
                target_accuracy_pct=98.0,
            )
        if _engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not ready.")
        return ModelInfoResponse(
            model_version=settings.model_version,
            inference_mode=settings.inference_mode,
            input_size=settings.input_size,
            num_classes=_engine.kb.num_trainable_classes,
            plant_guard_enabled=settings.plant_guard_enabled,
            tta_enabled=settings.tta_enabled,
            confidence_threshold=settings.confidence_threshold,
            confidence_margin=settings.confidence_margin,
            val_accuracy_pct=_load_val_accuracy_pct(),
            target_accuracy_pct=98.0,
        )

    @app.get("/v1/classes", response_model=ClassesListResponse)
    def list_classes() -> ClassesListResponse:
        settings = get_settings()
        kb = _engine.kb if _engine is not None else KnowledgeBase(settings.resolved_classes_path)
        classes = [
            ClassSummary(
                class_id=entry.class_id,
                type=entry.type,
                diseaseName=entry.diseaseName,
                diseaseNameRw=entry.diseaseNameRw,
            )
            for cid in kb.class_ids
            for entry in [kb.try_get(cid)]
            if entry is not None
        ]
        return ClassesListResponse(
            model_version=settings.model_version,
            count=len(classes),
            classes=classes,
        )

    @app.get("/v1/classes/{class_id}", response_model=ClassDetailResponse)
    def class_detail(class_id: str) -> ClassDetailResponse:
        settings = get_settings()
        kb = _engine.kb if _engine is not None else KnowledgeBase(settings.resolved_classes_path)
        entry = kb.try_get(class_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Class not found.")
        return ClassDetailResponse(
            class_id=class_id,
            result=kb.to_detection(entry, confidence_pct=0.0),
        )

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(file: UploadFile = File(...)) -> DetectResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Expected an image file (image/*).")

        raw = await file.read()
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

        settings = get_settings()
        mode = settings.inference_mode.lower().strip()

        if mode == "roboflow":
            try:
                return run_roboflow_detect(raw, settings=settings)
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Roboflow inference failed: {e}") from e

        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(raw)))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Could not decode image.")

        if _engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not ready.")

        return run_detect_with_engine(_engine, image)

    return app


app = create_app()
