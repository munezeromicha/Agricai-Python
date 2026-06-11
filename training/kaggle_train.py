#!/usr/bin/env python3
"""
AGRIC AI — Kaggle runner.
Dataset: /kaggle/input/datasets/giramatacandide/agricai-tomato-with-not-tomato
Scripts: /kaggle/input/datasets/giramatacandide/agricai-training-scripts

In notebook:  %run kaggle_train.py   (after copying to /kaggle/working/)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Exact Kaggle paths (giramatacandide)
DATASET_INPUT = Path(
    "/kaggle/input/datasets/giramatacandide/agricai-tomato-with-not-tomato"
)
SCRIPTS_INPUT = Path(
    "/kaggle/input/datasets/giramatacandide/agricai-training-scripts"
)
WORK_DIR = Path("/kaggle/working/model")
DATASET_WORK = Path("/kaggle/working/dataset")


def _resolve_dataset_root(base: Path) -> Path:
    """Find folder that directly contains train/ and val/."""
    if (base / "train").is_dir() and ((base / "val").is_dir() or (base / "validation").is_dir()):
        return base
    nested = base / "tomatoes leaf disease detection"
    if (nested / "train").is_dir():
        return nested
    for p in base.rglob("*"):
        if p.is_dir() and (p / "train").is_dir() and ((p / "val").is_dir() or (p / "validation").is_dir()):
            return p
    raise FileNotFoundError(f"No train/ + val/ found under {base}")


def _copy_dataset_to_working(src: Path) -> Path:
    if DATASET_WORK.exists():
        shutil.rmtree(DATASET_WORK)
    print(f"[setup] Copying dataset (writable copy for Not_Tomato flatten)...")
    print(f"        from: {src}")
    print(f"        to:   {DATASET_WORK}")
    shutil.copytree(src, DATASET_WORK)
    return DATASET_WORK


def _copy_scripts_to_working() -> None:
    for name in ("train_tomato.py", "train_tomato_gate.py", "kaggle_train.py"):
        src = SCRIPTS_INPUT / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, Path("/kaggle/working") / name)
        print(f"[setup] Copied {name}")


def main() -> None:
    src_root = _resolve_dataset_root(DATASET_INPUT)
    dataset = _copy_dataset_to_working(src_root)
    _copy_scripts_to_working()

    os.environ["TOMATO_DATASET_PATH"] = str(dataset)
    os.environ["AGRICAI_WORK_DIR"] = str(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/kaggle/working")
    os.chdir("/kaggle/working")

    from train_tomato import main as train_main

    print("=" * 72)
    print("AGRIC AI tomato training")
    print(f"Dataset input:  {DATASET_INPUT}")
    print(f"Dataset train:  {dataset}")
    print(f"Scripts input:  {SCRIPTS_INPUT}")
    print(f"Model output:   {WORK_DIR}")
    print("=" * 72)
    train_main()


if __name__ == "__main__":
    main()
