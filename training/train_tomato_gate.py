#!/usr/bin/env python3
"""
Stage 1 — train a binary tomato-leaf gate (tomato leaf vs everything else).

Positive class ``tomato_leaf``: all Tomato___* training images.
Negative class ``not_tomato``: images from a ``Not_Tomato/`` folder in your dataset.

Outputs (default: model/tomato_leaf_gate.keras):
  - tomato_leaf_gate.keras
  - tomato_gate_summary.json

Run standalone:
  python training/train_tomato_gate.py

Or it runs automatically at the end of train_tomato.py when a Not_Tomato folder exists.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 8
EPOCHS_FINE = 15
EARLY_STOP_PATIENCE = 4

GATE_CLASS_NAMES = ["not_tomato", "tomato_leaf"]
NOT_TOMATO_CLASS_ID = "Not_Tomato"
TOMATO_PREFIX = "Tomato___"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

random.seed(SEED)
np.random.seed(SEED)


def _import_tf():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    return tf, keras, layers


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.symlink(src.resolve(), dest)
    except OSError:
        shutil.copy2(src, dest)


def build_gate_split(
    train_dir: Path,
    val_dir: Path,
    gate_root: Path,
) -> tuple[Path, Path]:
    """Build binary folders: tomato_leaf (all Tomato___*) and not_tomato (Not_Tomato/)."""
    for subset, src_root in (("train", train_dir), ("val", val_dir)):
        pos_out = gate_root / subset / "tomato_leaf"
        neg_out = gate_root / subset / "not_tomato"
        if pos_out.exists():
            shutil.rmtree(pos_out)
        if neg_out.exists():
            shutil.rmtree(neg_out)
        pos_out.mkdir(parents=True, exist_ok=True)
        neg_out.mkdir(parents=True, exist_ok=True)

        for child in sorted(src_root.iterdir()):
            if not child.is_dir():
                continue
            images = [
                f for f in child.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not images:
                continue
            if child.name == NOT_TOMATO_CLASS_ID:
                dest_dir = neg_out
            elif child.name.startswith(TOMATO_PREFIX):
                dest_dir = pos_out
            else:
                continue
            for img in images:
                dest = dest_dir / f"{child.name}_{img.stem}{img.suffix}"
                if dest.exists():
                    dest = dest_dir / f"{child.name}_{img.stem}_{hash(img) & 0xFFFF}{img.suffix}"
                _link_or_copy(img, dest)

        pos_n = sum(1 for _ in pos_out.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)
        neg_n = sum(1 for _ in neg_out.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"[gate-data] {subset}: {pos_n} tomato_leaf, {neg_n} not_tomato")
        if pos_n == 0 or neg_n == 0:
            raise RuntimeError(
                f"Gate needs both tomato_leaf and not_tomato images in {subset}. "
                "Add a Not_Tomato/ folder with 500+ non-tomato photos."
            )

    return gate_root / "train", gate_root / "val"


def build_gate_datasets(tf, keras, layers, train_dir: Path, val_dir: Path):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.12),
        layers.RandomBrightness(0.10),
    ], name="gate_augment")

    def preprocess_train(image, label):
        image = augment(image, training=True)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, label

    def preprocess_val(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, label

    train_ds = keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        class_names=GATE_CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        str(val_dir),
        labels="inferred",
        class_names=GATE_CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
    )


def build_gate_model(keras, layers):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(2, activation=None, name="logits")(x)
    return keras.Model(inputs, outputs, name="tomato_leaf_gate"), base


def train_gate(model, base, train_ds, val_ds, keras):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1,
        ),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("\n=== Gate stage 1: frozen backbone ===")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks, verbose=1)

    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("\n=== Gate stage 2: fine-tune ===")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks, verbose=1)


def train_tomato_gate(
    train_dir: Path,
    val_dir: Path,
    *,
    work_dir: Path | None = None,
    gate_data_dir: Path | None = None,
) -> dict:
    """Train binary gate and save model/tomato_leaf_gate.keras."""
    tf, keras, layers = _import_tf()
    root = Path(__file__).resolve().parent.parent
    out_dir = work_dir or (root / "model")
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_data = gate_data_dir or (out_dir / "gate_split_data")

    gate_train, gate_val = build_gate_split(train_dir, val_dir, gate_data)
    train_ds, val_ds = build_gate_datasets(tf, keras, layers, gate_train, gate_val)
    model, base = build_gate_model(keras, layers)
    train_gate(model, base, train_ds, val_ds, keras)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    gate_path = out_dir / "tomato_leaf_gate.keras"
    model.save(gate_path)
    print(f"[gate] Saved → {gate_path}")

    summary = {
        "model": "tomato_leaf_gate",
        "class_names": GATE_CLASS_NAMES,
        "tomato_leaf_index": GATE_CLASS_NAMES.index("tomato_leaf"),
        "img_size": IMG_SIZE,
        "metrics": {"val_loss": float(val_loss), "val_accuracy": float(val_acc)},
        "deploy": {
            "TOMATO_GATE_ENABLED": True,
            "TOMATO_GATE_PATH": "model/tomato_leaf_gate.keras",
            "TOMATO_GATE_THRESHOLD": 0.55,
            "KERAS_PREPROCESS": "imagenet",
        },
    }
    summary_path = out_dir / "tomato_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[gate] Summary → {summary_path}")
    return summary


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_tomato import ensure_train_val, find_dataset_root

    root = find_dataset_root()
    train_dir, val_dir = ensure_train_val(root)
    if not (train_dir / NOT_TOMATO_CLASS_ID).is_dir() and not (val_dir / NOT_TOMATO_CLASS_ID).is_dir():
        raise SystemExit(
            "No Not_Tomato/ folder found. Add train/Not_Tomato/ with non-tomato images "
            "(maize leaves, faces, documents, random photos)."
        )
    work = Path(os.environ.get("AGRICAI_GATE_DIR", Path(__file__).resolve().parent.parent / "model"))
    summary = train_tomato_gate(train_dir, val_dir, work_dir=work)
    acc = summary["metrics"]["val_accuracy"]
    print(f"\nGate training complete — val accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
