from __future__ import annotations

import os

from app.inference.crops.banana import BANANA
from app.inference.crops.beans import BEANS
from app.inference.crops.cassava import CASSAVA
from app.inference.crops.coffee import COFFEE
from app.inference.crops.maize import MAIZE
from app.inference.crops.mango import MANGO
from app.inference.crops.onion import ONION
from app.inference.crops.orange import ORANGE
from app.inference.crops.potato import POTATO
from app.inference.crops.tea import TEA
from app.inference.crops.tomato import TOMATO
from app.inference.crops.types import CropConfig

_ALL_CROPS: tuple[CropConfig, ...] = (
    TOMATO,
    COFFEE,
    BEANS,
    TEA,
    CASSAVA,
    ONION,
    BANANA,
    MANGO,
    ORANGE,
    POTATO,
    MAIZE,
)

_BY_ID: dict[str, CropConfig] = {c.id: c for c in _ALL_CROPS}

DEFAULT_CROP_ID = "tomato"


def list_crops() -> list[CropConfig]:
    return list(_ALL_CROPS)


def get_crop(crop_id: str | None) -> CropConfig:
    key = (crop_id or DEFAULT_CROP_ID).strip().lower()
    if key not in _BY_ID:
        raise ValueError(f"Unknown crop '{crop_id}'. Supported: {', '.join(_BY_ID)}")
    return _BY_ID[key]


def resolve_workflow_id(crop: CropConfig) -> str:
    if not crop.workflow_id_env:
        raise ValueError(f"Crop {crop.id} is missing workflow_id_env.")
    workflow_id = os.environ.get(crop.workflow_id_env, "").strip()
    if not workflow_id:
        raise ValueError(
            f"{crop.workflow_id_env} is not set. Configure the Roboflow workflow id for {crop.display_name}."
        )
    return workflow_id
