# =============================================================================
# Agric AI — Full Kaggle training script (run once, top to bottom)
# =============================================================================
# Paste this entire file into ONE Kaggle notebook cell, or run as a script.
#
# Before running:
#   1. Notebook Settings → Accelerator → GPU (T4 or better)
#   2. Add dataset: giramatacandide/agricai-dataset
#
# Expected layout (matches AgricAI_Dataset on disk):
#   agricai-dataset/          (or AgricAI_Dataset/ inside it)
#     train/
#       Beans_healthy/
#       Tomato___Late_blight/
#       ...
#     validation/             (same class folder names as train)
#
# Outputs in /kaggle/working/:
#   - crop_classifier.keras
#   - crop_classifier.onnx
#   - class_names.json
#   - training_summary.json
#   - confusion_matrix.png (if matplotlib available)
#
# Classes are read from your dataset folders automatically (no fixed names in code).
# Do NOT add an "unknown" folder — Agricai-Python uses that when confidence is low.
# Production API uses ONNX + ImageNet preprocessing (see Agricai-Python README).
# =============================================================================

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Kaggle may mount datasets under different paths — we try all of these.
DATASET_CANDIDATES = [
    "/kaggle/input/datasets/giramatacandide/agricai-dataset",
    "/kaggle/input/agricai-dataset",
    "/kaggle/input/giramatacandide-agricai-dataset",
]

WORK_DIR = Path(os.environ.get("AGRICAi_WORK_DIR", "/kaggle/working"))
SPLIT_WORK_DIR = WORK_DIR / "split_data"  # built if dataset has no train/val

IMG_SIZE = 224
BATCH_SIZE = 32  # use 16 if GPU runs out of memory with many classes
SEED = 42
VAL_FRACTION = 0.2  # used only when we auto-split from one folder tree

EPOCHS_HEAD = 10  # frozen backbone
EPOCHS_FINE = 20  # fine-tune top layers
EARLY_STOP_PATIENCE = 5

# Filled automatically from dataset folder names (see discover_classes).
CLASS_NAMES: list[str] = []
NUM_CLASSES = 0

# Folders ignored during training (not crop classes)
SKIP_DIR_NAMES = frozenset({
    "unknown",
    "other",
    "misc",
    "background",
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "__macosx",
    ".ipynb_checkpoints",
})

# Same normalization as Agricai-Python/app/inference/engine.py (_preprocess_imagenet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------------------------------------------------
# Helpers — dataset discovery
# -----------------------------------------------------------------------------


def _set_class_names(names: list[str]) -> None:
    """Update global class list used by training (one folder = one class)."""
    global CLASS_NAMES, NUM_CLASSES
    CLASS_NAMES = names
    NUM_CLASSES = len(names)


def _is_skipped_dir(name: str) -> bool:
    n = name.lower().strip()
    return n in SKIP_DIR_NAMES or name.startswith(".")


def _class_dirs(split_dir: Path) -> list[str]:
    """Folder names under train/ or validation/ that contain images."""
    names: list[str] = []
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir() or _is_skipped_dir(child.name):
            continue
        if count_images_in_dir(child) > 0:
            names.append(child.name)
    return names


def _print_class_summary(classes: list[str]) -> None:
    print(f"[data] {len(classes)} classes (each subfolder = one label)")
    if len(classes) <= 8:
        for name in classes:
            print(f"       - {name}")
    else:
        for name in classes[:4]:
            print(f"       - {name}")
        print(f"       ... ({len(classes) - 7} more) ...")
        for name in classes[-3:]:
            print(f"       - {name}")


def discover_classes(train_dir: Path, val_dir: Path | None = None) -> list[str]:
    """
    Each immediate subfolder of train/ with images = one class.
    If val_dir is set, only classes present in BOTH train and validation are used.
    """
    train_classes = _class_dirs(train_dir)
    if not train_classes:
        raise RuntimeError(
            f"No class folders with images under {train_dir}. "
            "Expected: train/<class_name>/*.jpg"
        )

    if val_dir is None:
        _print_class_summary(train_classes)
        return train_classes

    val_set = set(_class_dirs(val_dir))
    classes = [c for c in train_classes if c in val_set]
    missing_in_val = [c for c in train_classes if c not in val_set]
    extra_in_val = sorted(val_set - set(train_classes))

    if missing_in_val:
        print(f"[warn] {len(missing_in_val)} class(es) in train/ but not in validation/ (skipped):")
        for name in missing_in_val[:5]:
            print(f"         {name}")
        if len(missing_in_val) > 5:
            print(f"         ... and {len(missing_in_val) - 5} more")
    if extra_in_val:
        print(f"[warn] {len(extra_in_val)} folder(s) only in validation/ (ignored for class list)")

    if not classes:
        raise RuntimeError(
            "No matching class folders between train/ and validation/. "
            "Use the same folder names in both splits."
        )

    _print_class_summary(classes)
    return classes


def resolve_split_root(candidate: Path) -> Path | None:
    """
    Return the directory that directly contains train/ (+ validation/).
    Handles: root/train/... or root/AgricAI_Dataset/train/...
    """
    if not candidate.is_dir():
        return None
    if (candidate / "train").is_dir():
        return candidate
    for child in sorted(candidate.iterdir()):
        if child.is_dir() and (child / "train").is_dir():
            print(f"[data] Found nested dataset folder: {child.name}")
            return child
    return None


def find_dataset_root() -> Path:
    for candidate in DATASET_CANDIDATES:
        resolved = resolve_split_root(Path(candidate))
        if resolved is not None:
            print(f"[data] Using dataset root: {resolved}")
            return resolved

    kaggle_in = Path("/kaggle/input")
    if kaggle_in.is_dir():
        for p in sorted(kaggle_in.rglob("*")):
            if not p.is_dir() or "agricai" not in p.name.lower():
                continue
            resolved = resolve_split_root(p)
            if resolved is not None:
                print(f"[data] Auto-found dataset root: {resolved}")
                return resolved

    raise FileNotFoundError(
        "Could not find train/ + validation/ under the Kaggle input path. "
        "Add giramatacandide/agricai-dataset and check Settings → Input for the exact path."
    )


def count_images_in_dir(directory: Path) -> int:
    n = 0
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            n += 1
    return n


def find_class_image_roots(root: Path) -> dict[str, Path]:
    """Find image folders under root/train (or root) when no train/val split exists."""
    search_roots: list[Path] = []
    if (root / "train").is_dir():
        search_roots.append(root / "train")
    elif (root / "Train").is_dir():
        search_roots.append(root / "Train")
    else:
        search_roots.append(root)

    found: dict[str, Path] = {}
    for base in search_roots:
        for dirpath, dirnames, filenames in os.walk(base):
            dp = Path(dirpath)
            parts_lower = {p.lower() for p in dp.parts}
            if parts_lower & {"val", "valid", "validation", "test"}:
                continue
            if not any(Path(f).suffix.lower() in IMAGE_EXTENSIONS for f in filenames):
                continue
            if _is_skipped_dir(dp.name):
                continue
            n = count_images_in_dir(dp)
            if n == 0:
                continue
            if dp.name not in found or n > count_images_in_dir(found[dp.name]):
                found[dp.name] = dp
    return found


def find_existing_split(root: Path) -> tuple[Path, Path] | None:
    """Return (train_dir, val_dir) — supports validation/ (your layout) and val/."""
    pairs = [
        (root / "train", root / "validation"),
        (root / "train", root / "val"),
        (root / "train", root / "valid"),
        (root / "Train", root / "Validation"),
        (root / "Train", root / "Val"),
    ]
    for train_p, val_p in pairs:
        if train_p.is_dir() and val_p.is_dir():
            if count_images_in_dir(train_p) > 0 and count_images_in_dir(val_p) > 0:
                print(f"[data] Using your split: {train_p.name}/ + {val_p.name}/")
                return train_p, val_p
    return None


def copy_split(class_to_paths: dict[str, Path], dest_train: Path, dest_val: Path) -> None:
    """Copy images into stratified train/val folders (symlinks on Linux/Kaggle)."""
    if dest_train.exists():
        shutil.rmtree(dest_train)
    if dest_val.exists():
        shutil.rmtree(dest_val)

    for cid, src_dir in class_to_paths.items():
        images = [
            f
            for f in src_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_FRACTION))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for subset, files in ((dest_train, train_images), (dest_val, val_images)):
            out_dir = subset / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in files:
                dest_file = out_dir / img_path.name
                if dest_file.exists():
                    dest_file = out_dir / f"{img_path.stem}_{hash(img_path) & 0xFFFF}{img_path.suffix}"
                try:
                    os.symlink(img_path.resolve(), dest_file)
                except OSError:
                    shutil.copy2(img_path, dest_file)

        print(
            f"[split] {cid}: {len(train_images)} train, {len(val_images)} val "
            f"(from {src_dir})"
        )


def ensure_train_val_dirs(root: Path) -> tuple[Path, Path]:
    existing = find_existing_split(root)
    if existing:
        train_p, val_p = existing
        _set_class_names(discover_classes(train_p, val_p))
        return train_p, val_p

    class_roots = find_class_image_roots(root)
    if not class_roots:
        print("[data] Folder tree under dataset root:")
        for dirpath, _, _ in os.walk(root):
            if len(Path(dirpath).relative_to(root).parts) <= 3:
                print(" ", dirpath)
        raise RuntimeError("No class folders with images found under the dataset.")

    train_p = SPLIT_WORK_DIR / "train"
    val_p = SPLIT_WORK_DIR / "val"
    print(f"[data] No train/val found — building split under {SPLIT_WORK_DIR}")
    copy_split(class_roots, train_p, val_p)
    _set_class_names(discover_classes(train_p, val_p))
    return train_p, val_p


# -----------------------------------------------------------------------------
# Preprocessing (matches Agricai-Python inference)
# -----------------------------------------------------------------------------

# Stronger augmentation — closer to phone photos (angle, light, blur, framing).
augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.12),
    ],
    name="augment",
)


def preprocess_train(image, label):
    image = augment(image, training=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, label


def preprocess_val(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, label


def build_datasets(train_dir: Path, val_dir: Path):
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
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def compute_class_weights(train_dir: Path) -> dict[int, float]:
    counts = Counter()
    for cid in CLASS_NAMES:
        class_dir = train_dir / cid
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        counts[cid] = n
    total = sum(counts.values())
    weights = {}
    for i, cid in enumerate(CLASS_NAMES):
        n = counts[cid] or 1
        weights[i] = total / (NUM_CLASSES * n)
    print("[data] Train counts:", dict(counts))
    print("[data] class_weight:", weights)
    return weights


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def build_model() -> keras.Model:
    """
    MobileNetV2 backbone + linear logits (no softmax).
    Input to this model must already be ImageNet-normalized (same as production ONNX).
    """
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation=None, name="logits")(x)
    model = keras.Model(inputs, outputs, name="agricai_crop_classifier")
    return model, base


# -----------------------------------------------------------------------------
# Training & evaluation
# -----------------------------------------------------------------------------


def train_model(model: keras.Model, base, train_ds, val_ds, class_weight: dict[int, float]):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("\n=== Stage 1: frozen backbone ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("\n=== Stage 2: fine-tune (top layers) ===")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    return history1, history2


def evaluate_model(model: keras.Model, val_ds) -> dict:
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[eval] val_loss={val_loss:.4f}  val_accuracy={val_acc:.4f}")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(logits, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    report = None
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
        )
        print("\n[eval] Classification report:\n")
        print(
            classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
        )

        cm = confusion_matrix(y_true, y_pred)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(WORK_DIR / "confusion_matrix.png", dpi=120)
            plt.close(fig)
            print(f"[eval] Saved confusion matrix → {WORK_DIR / 'confusion_matrix.png'}")
        except Exception as e:
            print(f"[eval] Could not save confusion matrix plot: {e}")
    except ImportError:
        print("[eval] sklearn not available — skipping detailed report")

    return {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "classification_report": report,
    }


# -----------------------------------------------------------------------------
# Export — .keras and .onnx for Agricai-Python
# -----------------------------------------------------------------------------


def save_artifacts(model: keras.Model, metrics: dict) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = WORK_DIR / "crop_classifier.keras"
    model.save(keras_path)
    print(f"\n[save] Keras model → {keras_path}")

    with open(WORK_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": CLASS_NAMES,
                "note": "Index i maps to trainable class_ids[i] in Agricai-Python (same order as data/classes.json, skip unknown row).",
                "num_trainable_classes": NUM_CLASSES,
                "unknown_folder_required": False,
            },
            f,
            indent=2,
        )

    summary = {
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "normalization": {
            "scale": "divide_by_255",
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
            "layout": "NCHW for ONNX in Agricai-Python",
        },
        "metrics": metrics,
        "deploy": {
            "INFERENCE_MODE": "onnx",
            "MODEL_PATH": "path/to/crop_classifier.onnx",
            "MODEL_VERSION": "1.0.0",
            "INPUT_SIZE": IMG_SIZE,
            "CONFIDENCE_THRESHOLD": 0.58,
            "CONFIDENCE_MARGIN": 0.10,
            "TTA_ENABLED": True,
            "onnx_output": f"logits shape [1, {NUM_CLASSES}]",
        },
    }
    with open(WORK_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Summary → {WORK_DIR / 'training_summary.json'}")

    export_onnx(model)


def export_onnx(model: keras.Model) -> None:
    import sys

    onnx_path = WORK_DIR / "crop_classifier.onnx"
    try:
        import onnx
        import tf2onnx
    except ImportError:
        print("[onnx] Installing tf2onnx + onnx …")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnx", "onnxruntime"]
        )
        import onnx
        import tf2onnx

    # Input: normalized float32 NHWC (batch, 224, 224, 3) — matches engine after preprocess
    spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, str(onnx_path))
    print(f"[onnx] Saved → {onnx_path}")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"[onnx] Verified input={inp.name} shape={inp.shape} dtype={inp.type}")
        print(f"[onnx] Verified output={out.name} shape={out.shape}")

        # dummy normalized tensor smoke test
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        logits = sess.run([out.name], {inp.name: dummy})[0]
        assert logits.shape == (1, NUM_CLASSES), f"Expected (1,{NUM_CLASSES}), got {logits.shape}"
        print("[onnx] Smoke test OK")
    except Exception as e:
        print(f"[onnx] Verification warning: {e}")


# -----------------------------------------------------------------------------
# Main — runs entire pipeline once
# -----------------------------------------------------------------------------


def main() -> None:
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    dataset_root = find_dataset_root()
    train_dir, val_dir = ensure_train_val_dirs(dataset_root)

    train_ds, val_ds = build_datasets(train_dir, val_dir)
    class_weight = compute_class_weights(train_dir)

    model, base = build_model()
    model.summary()

    train_model(model, base, train_ds, val_ds, class_weight)
    metrics = evaluate_model(model, val_ds)
    save_artifacts(model, metrics)

    print("\n" + "=" * 72)
    print("DONE. Download from Kaggle Output:")
    print("  - crop_classifier.keras")
    print("  - crop_classifier.onnx  → set MODEL_PATH in Agricai-Python .env")
    print("  - class_names.json")
    print("  - training_summary.json")
    print("\nDeploy Agricai-Python:")
    print("  INFERENCE_MODE=onnx")
    print("  MODEL_PATH=/path/to/crop_classifier.onnx")
    print("  MODEL_VERSION=1.0.0")
    print("\nNext steps for Agricai-Python:")
    print("  1. Copy class_names.json order into data/classes.json (one entry per class).")
    print("  2. Add a final 'unknown' row in JSON only — not a dataset folder.")
    print("  3. Deploy crop_classifier.onnx with INFERENCE_MODE=onnx")
    print("=" * 72)


if __name__ == "__main__":
    import sys

    main()
