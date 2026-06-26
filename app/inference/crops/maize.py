from app.inference.crops.types import CropConfig

MAIZE = CropConfig(
    id="maize",
    display_name="Maize",
    inference_kind="detect",
    model_id="dr.mangosteen/1",
    kb_prefix="Maize",
    model_version_label="roboflow-dr-mangosteen-maize",
    validation_keywords=("maize", "corn", "rust", "streak", "blight", "gray", "grey", "spot", "healthy", "leaf"),
)
