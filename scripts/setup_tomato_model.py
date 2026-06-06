"""
Point Agricai-Python at the tomato-only Keras model for testing.

  python scripts/setup_tomato_model.py

Verifies model/tomato_model.keras, syncs data/classes_tomato.json, updates .env.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERAS_PATH = ROOT / "model" / "tomato_model.keras"
NAMES_PATH = ROOT / "model" / "tomato_class_names.json"
CLASSES_OUT = ROOT / "data" / "classes_tomato.json"
ENV_PATH = ROOT / ".env"

# Reuse entry builder from sync_classes_json
sys.path.insert(0, str(ROOT / "scripts"))
from sync_classes_json import entry_for  # noqa: E402

UNKNOWN_ROW = {
    "class_id": "unknown",
    "type": "unknown",
    "diseaseName": "Uncertain diagnosis",
    "diseaseNameRw": "Ntibizwi neza",
    "explanation": (
        "The image could not be matched with high confidence. "
        "Try a clearer photo of the affected tomato leaf in natural light."
    ),
    "explanationRw": (
        "Ifoto ntiyahuje n'ukuri guhanitse. Gerageza ifoto yizewe y'ikibabi cy'inyanya mu mucyo wa kare."
    ),
    "treatment": "Seek local extension advice before applying chemicals.",
    "treatmentRw": "Shakisha inama y'impuguke mbere yo gukoresha imiti.",
    "prevention": "Use clean planting material and scout tomato fields weekly.",
    "preventionRw": "Koresha imbuto nbyo byiza kandi usuzume umurima w'inyanya buri cyumweru.",
    "care": "Monitor spread after rain; keep good airflow between plants.",
    "careRw": "Kurikirana nyuma y'imvura; komeza umuyaga hagati y'ibimera.",
}


def verify_keras(num_expected: int) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        print("[warn] tensorflow not installed — skipping model verification")
        return
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    last = model.layers[-1]
    units = getattr(last, "units", None)
    if units != num_expected:
        raise SystemExit(
            f"Model has {units} outputs but class_names.json has {num_expected}. "
            "Update model/tomato_class_names.json to match training order."
        )
    print(f"[ok] Keras model verified: {units} classes, last layer={last.name}")


def write_classes_json(names: list[str]) -> None:
    classes = [entry_for(n) for n in names]
    classes.append(UNKNOWN_ROW)
    CLASSES_OUT.write_text(
        json.dumps({"classes": classes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] Wrote {CLASSES_OUT} ({len(names)} tomato classes + unknown)")


def update_env() -> None:
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    updates = {
        "INFERENCE_MODE": "keras",
        "MODEL_PATH": "model/tomato_model.keras",
        "CLASSES_PATH": "data/classes_tomato.json",
        "MODEL_VERSION": "tomato-cnn-1.0.0",
        "KERAS_PREPROCESS": "builtin_rescale",
    }
    seen = set()
    new_lines = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.strip().startswith("#") else None
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[ok] Updated {ENV_PATH}")


def main() -> None:
    if not KERAS_PATH.is_file():
        src = ROOT / "docs" / "tomato_model.keras"
        if src.is_file():
            KERAS_PATH.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src, KERAS_PATH)
            print(f"[ok] Copied {src} → {KERAS_PATH}")
        else:
            raise SystemExit(f"Missing {KERAS_PATH}")

    raw = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    names: list[str] = raw["class_names"]
    verify_keras(len(names))
    write_classes_json(names)
    update_env()
    print("\nTomato-only mode ready. Restart the API:")
    print("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
