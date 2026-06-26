from app.inference.crops.types import CropConfig

POTATO = CropConfig(
    id="potato",
    display_name="Potato",
    inference_kind="detect",
    model_id="potato_leaf_disease/1",
    kb_prefix="Potato",
    model_version_label="roboflow-potato-leaf-disease",
    validation_keywords=("potato", "blight", "healthy", "leaf", "early", "late"),
)
