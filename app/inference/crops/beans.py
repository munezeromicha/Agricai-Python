from app.inference.crops.types import CropConfig

BEANS = CropConfig(
    id="beans",
    display_name="Beans",
    inference_kind="detect",
    model_id="bean-leaf/1",
    kb_prefix="Beans",
    model_version_label="roboflow-bean-leaf",
    validation_keywords=("bean", "angular", "rust", "healthy", "spot", "leaf"),
)
