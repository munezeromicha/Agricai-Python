from app.inference.crops.types import CropConfig

MANGO = CropConfig(
    id="mango",
    display_name="Mango",
    inference_kind="detect",
    model_id="demo2-fberl/1",
    kb_prefix="Mango",
    model_version_label="roboflow-mango-demo2",
    validation_keywords=("mango", "anthracnose", "healthy", "disease", "leaf", "powdery"),
)
