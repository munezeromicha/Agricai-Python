from app.inference.crops.types import CropConfig

TOMATO = CropConfig(
    id="tomato",
    display_name="Tomato",
    inference_kind="detect",
    model_id="tomato-disease-b518h/3",
    classes_path="data/classes_tomato.json",
    kb_prefix="Tomato",
    model_version_label="roboflow-tomato-disease-b518h",
    validation_keywords=("tomato", "blight", "septoria", "mosaic", "mold", "mite", "bacterial", "spot", "curl", "healthy", "target"),
)
