from app.inference.crops.types import CropConfig

COFFEE = CropConfig(
    id="coffee",
    display_name="Coffee",
    inference_kind="classify",
    model_id="coffee-leaf-diseases-classification/5",
    kb_prefix="Coffee",
    model_version_label="roboflow-coffee-leaf-diseases",
    validation_keywords=("coffee", "rust", "miner", "beetle", "leaf", "healthy", "disease", "cercospora"),
)
