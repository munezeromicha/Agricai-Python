#!/usr/bin/env python3
"""
Suggest CONFIDENCE_THRESHOLD / CONFIDENCE_MARGIN from a labeled image folder.

Folder layout (class name = subfolder name, must match class_id in classes.json):
  field_test/
    Tomato___Late_blight/
      img1.jpg
    Tomato___healthy/
      img2.jpg

Usage:
  python scripts/tune_thresholds.py field_test/
  python scripts/tune_thresholds.py field_test/ --no-tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from app.config import Settings, get_settings
from app.inference.engine import get_engine

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def load_labeled(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXT and f.is_file():
                pairs.append((f, label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Root with one subfolder per class_id")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()

    pairs = load_labeled(args.dataset_dir)
    if not pairs:
        print("No labeled images found (expect subfolders named like class_id).", file=sys.stderr)
        sys.exit(1)

    base = get_settings()
    settings = Settings(
        **{
            **base.model_dump(),
            "tta_enabled": not args.no_tta,
            "plant_guard_enabled": not args.no_guard,
        }
    )
    engine = get_engine(settings)
    trainable = set(engine.kb.trainable_class_ids)

    records: list[tuple[bool, float, float, str, str | None]] = []
    skipped = 0
    for path, label in pairs:
        if label not in trainable:
            skipped += 1
            continue
        img = Image.open(path)
        img.load()
        d = engine.classify(img)
        if d.plant_guard_blocked or d.probs.size == 0:
            records.append((False, 0.0, 0.0, label, None))
            continue
        top_i = int(np.argmax(d.probs))
        pred = engine.kb.trainable_class_ids[top_i]
        correct = pred == label
        records.append((correct, d.top_confidence, d.margin, label, pred))

    if not records:
        print("No valid labels matched trainable classes.", file=sys.stderr)
        sys.exit(1)

    print(f"Images: {len(records)}  (skipped unknown folders: {skipped})")
    print(f"TTA: {settings.tta_enabled}  plant_guard: {settings.plant_guard_enabled}\n")

    thresholds = [0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]
    margins = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]

    best = (0.0, 0.58, 0.10)
    best_score = -1.0

    print(f"{'thresh':>7} {'margin':>7} {'accept%':>8} {'acc|accept':>12}")
    for th in thresholds:
        for mg in margins:
            accepted = 0
            correct_accepted = 0
            for ok, conf, margin, _label, pred in records:
                uncertain = conf < th or margin < mg
                if not uncertain and pred is not None:
                    accepted += 1
                    if ok:
                        correct_accepted += 1
            accept_rate = accepted / len(records)
            acc_when = (correct_accepted / accepted) if accepted else 0.0
            score = acc_when * 0.7 + accept_rate * 0.3
            if score > best_score:
                best_score = score
                best = (score, th, mg)
            print(f"{th:7.2f} {mg:7.2f} {accept_rate:7.1%} {acc_when:11.1%}")

    print(f"\nSuggested (balance accuracy vs coverage): CONFIDENCE_THRESHOLD={best[1]:.2f}")
    print(f"  CONFIDENCE_MARGIN={best[2]:.2f}")
    print("Copy into .env and restart the vision API.")


if __name__ == "__main__":
    main()
