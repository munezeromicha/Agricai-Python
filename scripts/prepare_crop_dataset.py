#!/usr/bin/env python3
"""
Prepare a crop disease dataset for AGRIC AI training.

Takes a downloaded raw dataset (PlantVillage / Kaggle / Mendeley), optionally
renames class folders, splits into train/validation/test, and builds a
Not_<Crop>/ reject class from other-crop image folders (hard negatives).

Usage (from Agricai-Python root):

  python scripts/prepare_crop_dataset.py ^
    --crop Tomato ^
    --source D:\\datasets\\raw\\tomato ^
    --out D:\\datasets\\prepared\\tomato ^
    --negatives D:\\datasets\\raw\\maize D:\\datasets\\raw\\beans ^
    --val-fraction 0.15 --test-fraction 0.15

  # Optional rename map (JSON object: {"Old Folder": "Tomato___Early_blight", ...})
  python scripts/prepare_crop_dataset.py --crop Tomato --source ... --out ... --map rename.json

See docs/DATASET_SOURCES.md for which public datasets to download per crop.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
SKIP_DIR = {
    "unknown",
    "other",
    "misc",
    "background",
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "not_tomato",
    "not_maize",
    "not_bean",
}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


def discover_class_dirs(root: Path) -> dict[str, Path]:
    """
    Find class folders. Prefers root/train/* if present; otherwise immediate
    children of root that contain images (or nested PlantVillage color/ folders).
    """
    candidates: list[Path] = []
    for name in ("train", "Train", "color", "Color"):
        p = root / name
        if p.is_dir():
            candidates.append(p)
            break
    if not candidates:
        # Nested: root/*/train
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "train").is_dir():
                candidates.append(child / "train")
                break
    if not candidates:
        candidates.append(root)

    found: dict[str, Path] = {}
    for base in candidates:
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() in SKIP_DIR or child.name.startswith("."):
                continue
            if child.name.lower().startswith("not_"):
                continue
            n = len(list_images(child))
            if n == 0:
                continue
            found[child.name] = child
    return found


def collect_negative_images(negative_roots: list[Path], limit: int) -> list[Path]:
    images: list[Path] = []
    for root in negative_roots:
        if not root.exists():
            print(f"[warn] Negatives path missing: {root}")
            continue
        # Prefer class folders under train/, else all images under root
        class_dirs = discover_class_dirs(root)
        if class_dirs:
            for d in class_dirs.values():
                images.extend(list_images(d))
        else:
            images.extend(list_images(root))
    random.shuffle(images)
    if limit > 0:
        images = images[:limit]
    return images


def split_list(items: list[Path], val_f: float, test_f: float) -> tuple[list[Path], list[Path], list[Path]]:
    random.shuffle(items)
    n = len(items)
    n_test = max(1, int(n * test_f)) if test_f > 0 and n >= 10 else 0
    n_val = max(1, int(n * val_f)) if val_f > 0 and n >= 5 else 0
    if n_test + n_val >= n:
        n_test = max(0, n // 10)
        n_val = max(0, n // 10)
    test = items[:n_test]
    val = items[n_test : n_test + n_val]
    train = items[n_test + n_val :]
    if not train and items:
        train = items[:]
        val, test = [], []
    return train, val, test


def copy_into(files: list[Path], dest_dir: Path, *, link: bool) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in files:
        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{src.stem}_{abs(hash(str(src))) & 0xFFFF}{src.suffix}"
        try:
            if link:
                os_symlink = getattr(__import__("os"), "symlink")
                os_symlink(src.resolve(), dest)
            else:
                shutil.copy2(src, dest)
            written += 1
        except OSError:
            shutil.copy2(src, dest)
            written += 1
    return written


def load_rename_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--map JSON must be an object of old_name -> new_name")
    return {str(k): str(v) for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AGRIC AI crop dataset folders.")
    parser.add_argument("--crop", required=True, help="Crop name, e.g. Tomato, Maize, Bean")
    parser.add_argument("--source", type=Path, required=True, help="Raw downloaded dataset root")
    parser.add_argument("--out", type=Path, required=True, help="Output prepared dataset root")
    parser.add_argument(
        "--negatives",
        type=Path,
        nargs="*",
        default=[],
        help="Other-crop dataset roots used to build Not_<Crop>/",
    )
    parser.add_argument("--map", type=Path, default=None, help="Optional rename JSON")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--neg-limit", type=int, default=2000, help="Max images in Not_<Crop>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy (Linux/Kaggle)")
    parser.add_argument("--force", action="store_true", help="Delete existing --out if present")
    args = parser.parse_args()

    random.seed(args.seed)
    crop = args.crop.strip().replace(" ", "_")
    reject_name = f"Not_{crop}"

    if not args.source.is_dir():
        print(f"Source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    rename = load_rename_map(args.map)
    classes = discover_class_dirs(args.source)
    if not classes:
        print(f"No class folders with images under {args.source}", file=sys.stderr)
        sys.exit(1)

    if args.out.exists():
        if not args.force:
            print(f"Output exists: {args.out} (pass --force to overwrite)", file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(args.out)

    print(f"[prepare] crop={crop}")
    print(f"[prepare] source={args.source}")
    print(f"[prepare] out={args.out}")
    print(f"[prepare] found {len(classes)} class folders")

    summary: dict[str, dict[str, int]] = {}

    for old_name, src_dir in classes.items():
        new_name = rename.get(old_name, old_name)
        if new_name.lower().startswith("not_"):
            print(f"[skip] reject-like folder in source: {old_name}")
            continue
        images = list_images(src_dir)
        train, val, test = split_list(images, args.val_fraction, args.test_fraction)
        counts = {}
        for split, files in (("train", train), ("validation", val), ("test", test)):
            if not files:
                counts[split] = 0
                continue
            n = copy_into(files, args.out / split / new_name, link=args.symlink)
            counts[split] = n
        summary[new_name] = counts
        print(
            f"  {old_name} → {new_name}: "
            f"train={counts.get('train', 0)} val={counts.get('validation', 0)} "
            f"test={counts.get('test', 0)}"
        )

    # Hard negatives from other crops
    if args.negatives:
        neg_images = collect_negative_images(args.negatives, args.neg_limit)
        if not neg_images:
            print("[warn] No negative images found — Not_* folder not created")
        else:
            train, val, test = split_list(neg_images, args.val_fraction, args.test_fraction)
            counts = {}
            for split, files in (("train", train), ("validation", val), ("test", test)):
                if not files:
                    counts[split] = 0
                    continue
                n = copy_into(files, args.out / split / reject_name, link=args.symlink)
                counts[split] = n
            summary[reject_name] = counts
            print(
                f"  {reject_name}: train={counts.get('train', 0)} "
                f"val={counts.get('validation', 0)} test={counts.get('test', 0)} "
                f"(from {len(args.negatives)} negative root(s))"
            )
    else:
        print(
            f"[warn] No --negatives given. Add other crop leaves into "
            f"{reject_name}/ before training or the model will guess diseases on wrong plants."
        )

    meta = {
        "crop": crop,
        "reject_class": reject_name if reject_name in summary else None,
        "source": str(args.source.resolve()),
        "out": str(args.out.resolve()),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "classes": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "prepare_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[prepare] Wrote {meta_path}")
    print("[prepare] Next:")
    print(f"  1. Align data/classes_{crop.lower()}.json class_id with folder names")
    print("  2. Upload prepared folder to Kaggle (or set DATASET_PATH)")
    print("  3. Train with GPU — see docs/TRAIN_TOMATO.md / docs/DATASET_SOURCES.md")


if __name__ == "__main__":
    main()
