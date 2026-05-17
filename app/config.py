from functools import lru_cache
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

    # stub | onnx — stub needs no model file; onnx loads MODEL_PATH
    inference_mode: str = "stub"
    model_path: str | None = None
    model_version: str = "0.0.0-stub"

    # ImageNet-style preprocessing (change if your training recipe differs)
    input_size: int = 224

    # If max softmax probability is below this, return the "unknown" knowledge entry
    confidence_threshold: float = 0.35

    @property
    def classes_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        return root / "data" / "classes.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
