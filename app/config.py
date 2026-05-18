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

    # Reject prediction when top softmax score is below this (0–1). Use ~0.65+ to block
    # non-plant photos that still get a weak best-guess label (e.g. a house → 54% "healthy").
    confidence_threshold: float = 0.65

    # Top class must beat the runner-up by at least this margin (0–1), or we return unknown.
    confidence_margin: float = 0.18

    # Run color/contrast checks before ONNX — blocks terminals, houses, documents, etc.
    plant_guard_enabled: bool = True

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def classes_path(self) -> Path:
        return self.project_root / "data" / "classes.json"

    def resolved_model_path(self) -> Path | None:
        if not self.model_path:
            return None
        p = Path(self.model_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
