"""
Deploy models from Kaggle output zip into Agricai-Python.

  python scripts/deploy_kaggle_models.py
  python scripts/deploy_kaggle_models.py model/agricai_models.zip

Extracts essential files only (skips gate_split_data temp folders),
runs setup_tomato_model.py, and prints next steps.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "model" / "agricai_models.zip"
MODEL_DIR = ROOT / "model"

ESSENTIAL = {
    "tomato_classifier.keras": "tomato_model.keras",
    "tomato_class_names.json": "tomato_class_names.json",
    "tomato_leaf_gate.keras": "tomato_leaf_gate.keras",
    "tomato_gate_summary.json": "tomato_gate_summary.json",
    "tomato_training_summary.json": "tomato_training_summary.json",
}


def extract_zip(zip_path: Path) -> None:
    if not zip_path.is_file():
        raise SystemExit(f"Zip not found: {zip_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for src, dest_name in ESSENTIAL.items():
            try:
                data = zf.read(src)
            except KeyError:
                print(f"[skip] {src} not in zip")
                continue
            dest = MODEL_DIR / dest_name
            dest.write_bytes(data)
            print(f"[ok] {dest.name} ({len(data) // 1024} KB)")

    names_path = MODEL_DIR / "tomato_class_names.json"
    if names_path.is_file():
        meta = json.loads(names_path.read_text(encoding="utf-8"))
        print(f"[info] Classes: {meta.get('num_classes')} (Not_Tomato: {meta.get('has_not_tomato_class')})")

    summary = MODEL_DIR / "tomato_training_summary.json"
    if summary.is_file():
        m = json.loads(summary.read_text(encoding="utf-8")).get("metrics", {})
        print(f"[info] Disease val accuracy: {m.get('val_accuracy', 0) * 100:.2f}%")

    gate = MODEL_DIR / "tomato_gate_summary.json"
    if gate.is_file():
        m = json.loads(gate.read_text(encoding="utf-8")).get("metrics", {})
        print(f"[info] Gate val accuracy: {m.get('val_accuracy', 0) * 100:.2f}%")


def main() -> None:
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ZIP
    if not (MODEL_DIR / "tomato_model.keras").is_file() and zip_path.is_file():
        extract_zip(zip_path)
    elif (MODEL_DIR / "tomato_model.keras").is_file():
        print("[info] Model files already in model/ — running setup only")
    else:
        raise SystemExit(f"No models found. Place zip at {DEFAULT_ZIP} or extract manually.")

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "setup_tomato_model.py")])
    print("\nRestart the API:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
