from app.inference.crops.types import CropConfig

ORANGE = CropConfig(
    id="orange",
    display_name="Orange",
    inference_kind="detect",
    model_id="orange-hlb-disease/2",
    kb_prefix="Orange",
    model_version_label="roboflow-orange-hlb",
    validation_keywords=("orange", "hlb", "citrus", "huanglongbing", "greening", "healthy", "leaf"),
)
