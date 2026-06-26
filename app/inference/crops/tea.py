from app.inference.crops.types import CropConfig

TEA = CropConfig(
    id="tea",
    display_name="Tea",
    inference_kind="detect",
    model_id="tea-leaves-diseases/22",
    kb_prefix="Tea",
    model_version_label="roboflow-tea-leaves-diseases",
    validation_keywords=("tea", "blister", "red", "brown", "grey", "gray", "healthy", "leaf"),
)
