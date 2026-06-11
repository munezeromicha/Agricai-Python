#!/usr/bin/env python3
"""
Print top predictions and confidence-gate diagnostics for leaf images.

Usage (from Agricai-Python root, venv active):
  python scripts/debug_predict.py path/to/leaf.jpg
  python scripts/debug_predict.py path/to/folder --limit 20
  python scripts/debug_predict.py leaf.jpg --no-tta --no-guard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from app.config import Settings, get_settings
from app.inference.classify import prediction_is_uncertain
from app.inference.engine import get_engine
from app.inference.leaf_crop import extract_primary_leaf
from app.inference.plant_guard import leaf_plausibility_score, looks_like_crop_leaf_photo

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def collect_images(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXT and p.is_file()
    )
    return files[:limit] if limit > 0 else files


def analyze_one(path: Path, settings: Settings) -> None:
    engine = get_engine(settings)
    image = Image.open(path)
    image.load()

    work = image
    cropped = False
    if settings.leaf_auto_crop_enabled:
        work, cropped = extract_primary_leaf(image)
    leaf_score = leaf_plausibility_score(work)
    guard_ok = (
        looks_like_crop_leaf_photo(work, min_score=settings.plant_guard_min_score)
        if settings.plant_guard_enabled
        else True
    )
    details = engine.classify(image)

    print(f"\n{'=' * 72}")
    print(path)
    print(f"  auto_crop:   {'yes' if cropped else 'no'}")
    print(f"  leaf_score:  {leaf_score:.2f}")
    print(f"  plant_guard: {'pass' if guard_ok else 'BLOCKED'}")
    print(f"  tta_enabled: {settings.tta_enabled}")
    print(f"  threshold:   {settings.confidence_threshold:.0%}")
    print(f"  margin:      {settings.confidence_margin:.0%}")

    if details.plant_guard_blocked:
        print("  => REJECTED (plant_guard)")
        return

    uncertain, top_p, margin, reason = prediction_is_uncertain(
        details.probs,
        min_confidence=settings.confidence_threshold,
        min_margin=settings.confidence_margin,
    )
    verdict = "ACCEPTED" if not uncertain and details.top_class_id else f"UNCERTAIN ({reason})"
    print(f"  top_conf:    {top_p:.1%}  margin: {margin:.1%}")
    print(f"  => {verdict}" + (f" → {details.top_class_id}" if details.top_class_id else ""))

    print("  top-5:")
    for alt in details.alternatives[:5]:
        print(f"    {alt.confidence_pct:5.1f}%  {alt.class_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug crop classifier on image(s).")
    parser.add_argument("path", type=Path, help="Image file or folder")
    parser.add_argument("--limit", type=int, default=0, help="Max images when path is a folder")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--no-guard", action="store_true", help="Disable plant guard")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    settings = Settings(
        **{
            **settings.model_dump(),
            "tta_enabled": not args.no_tta,
            "plant_guard_enabled": not args.no_guard,
        }
    )

    paths = collect_images(args.path, args.limit)
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {settings.inference_mode}  version={settings.model_version}")
    for p in paths:
        try:
            analyze_one(p, settings)
        except Exception as e:
            print(f"\n{p}: ERROR — {e}")


if __name__ == "__main__":
    main()
