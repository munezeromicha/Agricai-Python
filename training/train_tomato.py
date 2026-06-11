#!/usr/bin/env python3
"""
Train a tomato-only leaf disease classifier for AGRIC AI.

Works on:
  - Kaggle notebook (GPU) with dataset "tomatoes-leaf-disease-detection"
  - Local machine: set TOMATO_DATASET_PATH to your dataset root

Expected dataset layout (any of these):

  Option A — pre-split (recommended):
    tomatoes-leaf-disease-detection/
      train/
        Tomato___Early_blight/
        Tomato___Late_blight/
        Tomato___healthy/
        ...
      validation/   (or val/)

  Option B — single tree (auto 80/20 split):
    tomatoes-leaf-disease-detection/
      Tomato___Early_blight/
      Tomato___Late_blight/
      ...

  Option C — nested (common on Kaggle):
    tomatoes-leaf-disease-detection/
      train/
        Tomato___Early_blight/
        ...

Outputs (in AGRICAI_WORK_DIR or ./model/tomato/):
  - tomato_classifier.keras
  - tomato_classifier.onnx
  - tomato_class_names.json
  - tomato_training_summary.json
  - tomato_confusion_matrix.png

Install (training machine / Kaggle):
  pip install tensorflow scikit-learn matplotlib tf2onnx onnx onnxruntime

Kaggle quick start:
  1. New notebook → Settings → Accelerator → GPU T4
  2. Add dataset: search "tomato leaf disease" (or your tomatoes-leaf-disease-detection zip)
  3. Paste this file or:  %run training/train_tomato.py
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
VAL_FRACTION = 0.2
EPOCHS_HEAD = 12
EPOCHS_FINE = 25
EARLY_STOP_PATIENCE = 6

# Where to write models (Kaggle: /kaggle/working/tomato_model)
WORK_DIR = Path(os.environ.get("AGRICAI_WORK_DIR", "tomato_model"))

# Kaggle / local dataset search paths
DATASET_CANDIDATES = [
    os.environ.get("TOMATO_DATASET_PATH", "").strip(),
    "/kaggle/input/tomatoes-leaf-disease-detection",
    "/kaggle/input/datasets/tomatoes-leaf-disease-detection",
    "/kaggle/input/tomato-leaf-disease-detection",
    "/kaggle/input/datasets/*/tomatoes-leaf-disease-detection",
    "./datasets/tomatoes-leaf-disease-detection",
    "../datasets/tomatoes-leaf-disease-detection",
]

SPLIT_WORK_DIR = WORK_DIR / "split_data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
SKIP_DIR_NAMES = frozenset({
    "unknown", "other", "misc", "background",
    "train", "val", "valid", "validation", "test",
    "__macosx", ".ipynb_checkpoints",
})

# 11th class — non-tomato images (faces, other crops, documents, etc.)
NOT_TOMATO_CLASS_ID = "Not_Tomato"
NOT_TOMATO_ALIASES = frozenset({
    "not_tomato", "not tomato", "not-tomato", "non_tomato", "non tomato",
    "negative", "negatives", "background", "other_crop", "other crop",
    "not_a_tomato", "not a tomato",
})

# ImageNet normalization — must match Agricai-Python inference (engine.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Map messy folder names → AGRIC AI class_id (matches data/classes.json)
CLASS_NAME_ALIASES: dict[str, str] = {
    "early blight": "Tomato___Early_blight",
    "early_blight": "Tomato___Early_blight",
    "tomato early blight": "Tomato___Early_blight",
    "tomato___early_blight": "Tomato___Early_blight",
    "late blight": "Tomato___Late_blight",
    "late_blight": "Tomato___Late_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato___late_blight": "Tomato___Late_blight",
    "yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato___tomato_yellow_leaf_curl_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato___tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "bacterial spot": "Tomato___Bacterial_spot",
    "tomato bacterial spot": "Tomato___Bacterial_spot",
    "tomato___bacterial_spot": "Tomato___Bacterial_spot",
    "septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato___septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "spider mites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "spider_mites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "target spot": "Tomato___Target_Spot",
    "tomato___target_spot": "Tomato___Target_Spot",
    "leaf mold": "Tomato___Leaf_Mold",
    "tomato___leaf_mold": "Tomato___Leaf_Mold",
    "healthy": "Tomato___healthy",
    "tomato healthy": "Tomato___healthy",
    "tomato___healthy": "Tomato___healthy",
}

CLASS_NAMES: list[str] = []
NUM_CLASSES = 0

random.seed(SEED)
np.random.seed(SEED)


def _import_tf():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    return tf, keras, layers


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


def normalize_class_name(raw: str) -> str:
    """Convert folder name to AGRIC AI Tomato___* class_id or Not_Tomato."""
    key = raw.strip().lower().replace("-", " ").replace("__", "_")
    key = re.sub(r"\s+", " ", key)
    key_us = key.replace(" ", "_")
    if key in NOT_TOMATO_ALIASES or key_us in NOT_TOMATO_ALIASES or raw.strip() == NOT_TOMATO_CLASS_ID:
        return NOT_TOMATO_CLASS_ID
    if key in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[key]
    # Already looks like Tomato___Something
    if raw.startswith("Tomato___"):
        return raw
    # PlantVillage style: Tomato___Late_blight
    cleaned = raw.replace(" ", "_").replace("-", "_")
    if cleaned.lower().startswith("tomato"):
        parts = cleaned.split("___", 1)
        if len(parts) == 2:
            return f"Tomato___{parts[1]}"
        return f"Tomato___{cleaned.split('Tomato_', 1)[-1]}"
    return f"Tomato___{cleaned}"


def _is_skipped_dir(name: str) -> bool:
    return name.lower().strip() in SKIP_DIR_NAMES or name.startswith(".")


def count_images(directory: Path) -> int:
    return sum(
        1 for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _set_classes(names: list[str]) -> None:
    global CLASS_NAMES, NUM_CLASSES
    CLASS_NAMES = _order_class_names(names)
    NUM_CLASSES = len(CLASS_NAMES)


def _order_class_names(names: list[str]) -> list[str]:
    """Tomato disease classes first (sorted), Not_Tomato always last."""
    tomato = sorted(cid for cid in names if cid != NOT_TOMATO_CLASS_ID)
    if NOT_TOMATO_CLASS_ID in names:
        return tomato + [NOT_TOMATO_CLASS_ID]
    return tomato


def has_not_tomato_class(train_dir: Path, val_dir: Path) -> bool:
    return (train_dir / NOT_TOMATO_CLASS_ID).is_dir() or (val_dir / NOT_TOMATO_CLASS_ID).is_dir()


def find_dataset_root() -> Path:
    for candidate in DATASET_CANDIDATES:
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_dir():
            if (p / "train").is_dir() or any(
                c.is_dir() and count_images(c) > 0
                for c in p.iterdir()
                if c.is_dir() and not _is_skipped_dir(c.name)
            ):
                print(f"[data] Using dataset: {p.resolve()}")
                return p

    # Kaggle auto-search
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.is_dir():
        for p in sorted(kaggle_in.rglob("*")):
            if not p.is_dir():
                continue
            name = p.name.lower()
            if "tomato" in name and ("disease" in name or "leaf" in name):
                if (p / "train").is_dir() or count_images(p) > 50:
                    print(f"[data] Auto-found: {p}")
                    return p

    raise FileNotFoundError(
        "Tomato dataset not found. Set TOMATO_DATASET_PATH or add "
        "tomatoes-leaf-disease-detection on Kaggle."
    )


def find_existing_split(root: Path) -> tuple[Path, Path] | None:
    pairs = [
        (root / "train", root / "validation"),
        (root / "train", root / "val"),
        (root / "Train", root / "Validation"),
    ]
    for train_p, val_p in pairs:
        if train_p.is_dir() and val_p.is_dir() and count_images(train_p) > 0:
            return train_p, val_p
    return None


def discover_class_dirs(split_dir: Path) -> dict[str, Path]:
    """Return {normalized_class_id: folder_path}."""
    found: dict[str, Path] = {}
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir() or _is_skipped_dir(child.name):
            continue
        if count_images(child) == 0:
            continue
        cid = normalize_class_name(child.name)
        found[cid] = child
    return found


def flatten_nested_class_folder(class_dir: Path) -> int:
    """
    TensorFlow only reads images directly inside a class folder.
    If Not_Tomato/bean_leaves/*.jpg etc. exist, link them into Not_Tomato/ root.
    """
    if not class_dir.is_dir():
        return 0
    direct = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if direct:
        return 0
    nested = [
        f for f in class_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.parent != class_dir
    ]
    if not nested:
        return 0
    linked = 0
    for img in nested:
        sub = img.parent.name
        dest = class_dir / f"{sub}_{img.stem}{img.suffix}"
        if dest.exists():
            dest = class_dir / f"{sub}_{img.stem}_{hash(img) & 0xFFFF}{img.suffix}"
        try:
            os.symlink(img.resolve(), dest)
        except OSError:
            shutil.copy2(img, dest)
        linked += 1
    print(f"[data] Flattened {linked} nested images in {class_dir.name}/")
    return linked


def prepare_dataset_folders(train_dir: Path, val_dir: Path) -> None:
    """Ensure nested Not_Tomato subfolders (bean_leaves, faces, …) are visible to TensorFlow."""
    for split in (train_dir, val_dir):
        nt = split / NOT_TOMATO_CLASS_ID
        if nt.is_dir():
            flatten_nested_class_folder(nt)


def copy_split(class_dirs: dict[str, Path], dest_train: Path, dest_val: Path) -> None:
    if dest_train.exists():
        shutil.rmtree(dest_train)
    if dest_val.exists():
        shutil.rmtree(dest_val)

    for cid, src_dir in class_dirs.items():
        images = [
            f for f in src_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_FRACTION))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for subset, files in ((dest_train, train_images), (dest_val, val_images)):
            out_dir = subset / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in files:
                dest = out_dir / img.name
                if dest.exists():
                    dest = out_dir / f"{img.stem}_{hash(img) & 0xFFFF}{img.suffix}"
                try:
                    os.symlink(img.resolve(), dest)
                except OSError:
                    shutil.copy2(img, dest)

        print(f"[split] {cid}: {len(train_images)} train, {len(val_images)} val")


def ensure_train_val(root: Path) -> tuple[Path, Path]:
    existing = find_existing_split(root)
    if existing:
        train_p, val_p = existing
        # Rebuild with normalized folder names if needed
        train_classes = discover_class_dirs(train_p)
        val_classes = discover_class_dirs(val_p)
        common = sorted(set(train_classes) & set(val_classes))
        if not common:
            raise RuntimeError("No matching classes between train/ and validation/")
        _set_classes(common)
        tomato_n = sum(1 for c in CLASS_NAMES if c != NOT_TOMATO_CLASS_ID)
        extra = f" + {NOT_TOMATO_CLASS_ID}" if NOT_TOMATO_CLASS_ID in CLASS_NAMES else ""
        print(f"[data] {tomato_n} tomato classes{extra}:")
        for c in CLASS_NAMES:
            print(f"       - {c}")
        return train_p, val_p

    class_dirs = discover_class_dirs(root)
    if not class_dirs and (root / "train").is_dir():
        class_dirs = discover_class_dirs(root / "train")

    if not class_dirs:
        raise RuntimeError(f"No tomato class folders with images under {root}")

    train_p = SPLIT_WORK_DIR / "train"
    val_p = SPLIT_WORK_DIR / "val"
    print(f"[data] Building {VAL_FRACTION:.0%} validation split → {SPLIT_WORK_DIR}")
    copy_split(class_dirs, train_p, val_p)
    _set_classes(sorted(class_dirs.keys()))
    return train_p, val_p


# -----------------------------------------------------------------------------
# Model & training
# -----------------------------------------------------------------------------


def build_datasets(tf, keras, layers, train_dir: Path, val_dir: Path):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.12),
    ], name="augment")

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
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        str(val_dir),
        labels="inferred",
        class_names=CLASS_NAMES,
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


def class_weights(train_dir: Path) -> dict[int, float]:
    counts = Counter()
    for cid in CLASS_NAMES:
        d = train_dir / cid
        counts[cid] = sum(
            1 for f in d.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ) if d.is_dir() else 0
    total = sum(counts.values()) or 1
    weights = {i: total / (NUM_CLASSES * (counts[c] or 1)) for i, c in enumerate(CLASS_NAMES)}
    print("[data] Train counts:", dict(counts))
    return weights


def build_model(keras, layers):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(NUM_CLASSES, activation=None, name="logits")(x)
    return keras.Model(inputs, outputs, name="tomato_classifier"), base


def train(model, base, train_ds, val_ds, cw, keras):
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
    print("\n=== Stage 1: frozen MobileNetV2 backbone ===")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              class_weight=cw, callbacks=callbacks, verbose=1)

    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("\n=== Stage 2: fine-tune top layers ===")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
              class_weight=cw, callbacks=callbacks, verbose=1)


def evaluate(model, val_ds) -> dict:
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[eval] val_accuracy = {val_acc:.4f} ({val_acc * 100:.2f}%)")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(logits, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    report = None
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True,
        )
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_yticks(range(NUM_CLASSES))
            short = [c.replace("Tomato___", "") for c in CLASS_NAMES]
            ax.set_xticklabels(short, rotation=45, ha="right")
            ax.set_yticklabels(short)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            fig.tight_layout()
            fig.savefig(WORK_DIR / "tomato_confusion_matrix.png", dpi=120)
            plt.close(fig)
            print(f"[eval] Confusion matrix → {WORK_DIR / 'tomato_confusion_matrix.png'}")
        except Exception as e:
            print(f"[eval] Plot skipped: {e}")
    except ImportError:
        print("[eval] pip install scikit-learn for classification report")

    return {"val_loss": float(val_loss), "val_accuracy": float(val_acc), "classification_report": report}


def export_onnx(model, keras) -> None:
    import tensorflow as tf

    onnx_path = WORK_DIR / "tomato_classifier.onnx"
    try:
        import onnx
        import tf2onnx
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnx"])
        import onnx
        import tf2onnx

    spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, str(onnx_path))
    print(f"[onnx] Saved → {onnx_path}")


def save_artifacts(model, metrics: dict) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = WORK_DIR / "tomato_classifier.keras"
    model.save(keras_path)
    print(f"[save] Keras → {keras_path}")

    with open(WORK_DIR / "tomato_class_names.json", "w", encoding="utf-8") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "crop": "tomato",
            "num_classes": NUM_CLASSES,
            "has_not_tomato_class": NOT_TOMATO_CLASS_ID in CLASS_NAMES,
            "note": "Index i → CLASS_NAMES[i]. Not_Tomato (if present) is always last.",
        }, f, indent=2)

    summary = {
        "model": "tomato_classifier",
        "crop": "tomato",
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "normalization": {
            "scale": "divide_by_255",
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
        },
        "metrics": metrics,
        "target_accuracy": 0.98,
        "deploy": {
            "INFERENCE_MODE": "onnx",
            "MODEL_PATH": "model/tomato/tomato_classifier.onnx",
            "MODEL_VERSION": "tomato-1.0.0",
            "INPUT_SIZE": IMG_SIZE,
        },
    }
    with open(WORK_DIR / "tomato_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    export_onnx(model, None)


def main() -> None:
    tf, keras, layers = _import_tf()
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    root = find_dataset_root()
    train_dir, val_dir = ensure_train_val(root)
    prepare_dataset_folders(train_dir, val_dir)
    train_ds, val_ds = build_datasets(tf, keras, layers, train_dir, val_dir)
    cw = class_weights(train_dir)

    model, base = build_model(keras, layers)
    model.summary()

    train(model, base, train_ds, val_ds, cw, keras)
    metrics = evaluate(model, val_ds)
    save_artifacts(model, metrics)

    if has_not_tomato_class(train_dir, val_dir):
        print("\n[data] Not_Tomato folder found — training Stage 1 binary gate...")
        from train_tomato_gate import train_tomato_gate

        gate_dir = WORK_DIR.parent if WORK_DIR.name != "model" else WORK_DIR
        train_tomato_gate(train_dir, val_dir, work_dir=gate_dir)
    else:
        print(
            "\n[hint] Add train/Not_Tomato/ with non-tomato images (500+) to train "
            "the Stage 1 gate and enable the 11th reject class."
        )

    acc = metrics["val_accuracy"]
    print("\n" + "=" * 72)
    print(f"TOMATO MODEL TRAINING COMPLETE — val accuracy: {acc * 100:.2f}%")
    print(f"Artifacts in: {WORK_DIR.resolve()}")
    print("  - tomato_classifier.keras")
    print("  - tomato_classifier.onnx")
    print("  - tomato_class_names.json")
    print("  - tomato_training_summary.json")
    if has_not_tomato_class(train_dir, val_dir):
        print("  - tomato_leaf_gate.keras (Stage 1 binary gate)")
        print("  - tomato_gate_summary.json")
    if acc >= 0.98:
        print("✓ Reached 98% validation accuracy target!")
    else:
        print("→ To reach 98%: add more field photos, train longer, or balance classes.")
    print("=" * 72)


if __name__ == "__main__":
    main()
