from app.inference.crops.types import CropConfig

BANANA = CropConfig(
    id="banana",
    display_name="Banana",
    inference_kind="detect",
    model_id="banana-leaf-disease-qesr2-m44h9/1",
    kb_prefix="Banana",
    model_version_label="roboflow-banana-leaf-disease",
    validation_keywords=("banana", "sigatoka", "yellow", "healthy", "leaf", "disease"),
)
