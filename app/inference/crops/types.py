from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InferenceKind = Literal["detect", "classify", "workflow"]


@dataclass(frozen=True)
class CropConfig:
    """Roboflow-backed crop definition."""

    id: str
    display_name: str
    inference_kind: InferenceKind
    model_id: str | None = None
    classes_path: str | None = None
    kb_prefix: str | None = None
    workflow_workspace: str | None = None
    workflow_id_env: str | None = None
    workflow_classes: str | None = None
    model_version_label: str = ""
    # Lowercase substrings expected in Roboflow class names for this crop (wrong-crop guard).
    validation_keywords: tuple[str, ...] = ()

    def resolved_model_id(self) -> str:
        if not self.model_id:
            raise ValueError(f"Crop {self.id} has no Roboflow model id.")
        return self.model_id.strip("/")

    def resolved_classes_path(self, project_root) -> str:
        from pathlib import Path

        if self.classes_path:
            p = Path(self.classes_path)
            if not p.is_absolute():
                p = project_root / p
            return str(p)
        return str(project_root / "data" / "classes.json")
