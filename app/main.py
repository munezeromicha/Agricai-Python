from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from PIL import UnidentifiedImageError

from app.config import get_settings
from app.inference.engine import InferenceEngine, get_engine, run_detect_with_engine
from app.schemas import DetectResponse, HealthResponse, ModelInfoResponse

_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
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
        return HealthResponse(
            status="ok",
            model_version=settings.model_version,
            inference_mode=settings.inference_mode,
        )

    @app.get("/v1/models", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
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
        )

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(file: UploadFile = File(...)) -> DetectResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Expected an image file (image/*).")

        raw = await file.read()
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

        try:
            image = Image.open(BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Could not decode image.")

        if _engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not ready.")

        return run_detect_with_engine(_engine, image)

    return app


app = create_app()
