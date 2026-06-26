from app.inference.crops.types import CropConfig

ONION = CropConfig(
    id="onion",
    display_name="Onion",
    inference_kind="detect",
    model_id="onion-disease-hmx1h/2",
    kb_prefix="Onion",
    model_version_label="roboflow-onion-disease",
    validation_keywords=("onion", "purple", "blotch", "mildew", "rot", "healthy", "leaf"),
)
