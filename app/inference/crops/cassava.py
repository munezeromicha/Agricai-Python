from app.inference.crops.types import CropConfig

CASSAVA = CropConfig(
    id="cassava",
    display_name="Cassava",
    inference_kind="detect",
    model_id="cassava-leaves-disease/5",
    kb_prefix="Cassava",
    model_version_label="roboflow-cassava-leaves-disease",
    validation_keywords=("cassava", "mosaic", "mottle", "blight", "bacterial", "healthy"),
)
