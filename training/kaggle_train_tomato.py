"""
Enhanced tomato leaf disease training — paste into ONE Kaggle notebook cell.

Dataset: giramatacandide/tomatoes-leaf-disease-detection
Settings: Notebook → Accelerator → GPU T4

Improvements over basic CNN:
  - MobileNetV2 transfer learning (much higher accuracy)
  - ImageNet normalization (matches Agricai-Python production inference)
  - Data augmentation for field photos
  - Class weights for imbalanced classes
  - Two-stage training: frozen backbone → fine-tune
  - Early stopping + learning-rate schedule
  - Classification report + confusion matrix
  - Exports .keras, .onnx, class_names.json, training_summary.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================================================================
# DATASET PATHS
# =============================================================================

DATASET_PATH = "/kaggle/input/datasets/giramatacandide/tomatoes-leaf-disease-detection"
WORK_DIR = Path("/kaggle/working")

train_dir = os.path.join(DATASET_PATH, "train")
val_dir = os.path.join(DATASET_PATH, "val")

# Fallback if dataset uses validation/ instead of val/
if not os.path.isdir(val_dir):
    val_dir = os.path.join(DATASET_PATH, "validation")

# =============================================================================
# SETTINGS
# =============================================================================

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 12   # frozen backbone
EPOCHS_FINE = 20   # fine-tune top layers
EARLY_STOP_PATIENCE = 5

# ImageNet stats — MUST match Agricai-Python inference (engine.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Normalize folder names → AGRIC AI class_id (Tomato___*)
CLASS_ALIASES = {
    "early blight": "Tomato___Early_blight",
    "late blight": "Tomato___Late_blight",
    "healthy": "Tomato___healthy",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "bacterial spot": "Tomato___Bacterial_spot",
    "septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "leaf mold": "Tomato___Leaf_Mold",
    "target spot": "Tomato___Target_Spot",
}


def normalize_class_name(raw: str) -> str:
    if raw.startswith("Tomato___"):
        return raw
    key = raw.lower().replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    cleaned = raw.replace(" ", "_").replace("-", "_")
    return cleaned if cleaned.startswith("Tomato") else f"Tomato___{cleaned}"


def rename_class_folders(split_dir: str) -> list[str]:
    """Rename subfolders to AGRIC AI convention; return sorted class list."""
    renamed: list[str] = []
    for name in sorted(os.listdir(split_dir)):
        src = os.path.join(split_dir, name)
        if not os.path.isdir(src):
            continue
        cid = normalize_class_name(name)
        dst = os.path.join(split_dir, cid)
        if src != dst and not os.path.exists(dst):
            os.rename(src, dst)
        elif src != dst:
            # merge: move images into existing folder
            for f in os.listdir(src):
                os.rename(os.path.join(src, f), os.path.join(dst, f))
            os.rmdir(src)
        renamed.append(cid)
    return sorted(set(renamed))


# =============================================================================
# PREPARE CLASS NAMES
# =============================================================================

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

class_names = rename_class_folders(train_dir)
rename_class_folders(val_dir)
# Keep only classes present in both splits
val_names = set(os.listdir(val_dir))
class_names = [c for c in class_names if c in val_names]
num_classes = len(class_names)

print("\nClasses Found:")
for cls in class_names:
    print(f"  - {cls}")
print(f"\nTotal Classes: {num_classes}")

# =============================================================================
# DATA AUGMENTATION + PREPROCESSING
# =============================================================================

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.06, 0.06),
    layers.RandomContrast(0.12),
    layers.RandomBrightness(0.10),
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


# =============================================================================
# LOAD DATASETS
# =============================================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    class_names=class_names,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    class_names=class_names,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =============================================================================
# CLASS WEIGHTS (handle imbalanced classes)
# =============================================================================

counts = Counter()
for cid in class_names:
    d = os.path.join(train_dir, cid)
    counts[cid] = sum(
        1 for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ) if os.path.isdir(d) else 0

total = sum(counts.values()) or 1
class_weight = {
    i: total / (num_classes * (counts[c] or 1))
    for i, c in enumerate(class_names)
}
print("\nTrain image counts:", dict(counts))
print("Class weights:", class_weight)

# =============================================================================
# BUILD MODEL — MobileNetV2 transfer learning (better than basic CNN)
# =============================================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.35, name="dropout")(x)
outputs = layers.Dense(num_classes, activation=None, name="logits")(x)  # no softmax — from_logits loss

model = models.Model(inputs, outputs, name="tomato_classifier")
model.summary()

# =============================================================================
# CALLBACKS
# =============================================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# =============================================================================
# STAGE 1 — Train classifier head (frozen backbone)
# =============================================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("\n" + "=" * 60)
print("STAGE 1: Frozen MobileNetV2 backbone")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# =============================================================================
# STAGE 2 — Fine-tune top layers
# =============================================================================

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("\n" + "=" * 60)
print("STAGE 2: Fine-tune top 50 MobileNetV2 layers")
print("=" * 60)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# Merge histories for plotting
history = {
    "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss": history1.history["loss"] + history2.history["loss"],
    "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
}

# =============================================================================
# EVALUATE MODEL
# =============================================================================

loss, accuracy = model.evaluate(val_ds, verbose=0)

print("\n" + "=" * 40)
print(f"Validation Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 40)

# Detailed classification report
y_true, y_pred = [], []
for images, labels in val_ds:
    logits = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(logits, axis=1).tolist())
    y_true.extend(labels.numpy().tolist())

classification_report_dict = None
try:
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    classification_report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    short_names = [c.replace("Tomato___", "") for c in class_names]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(WORK_DIR / "tomato_confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"\nConfusion matrix saved → {WORK_DIR / 'tomato_confusion_matrix.png'}")
except ImportError:
    print("[warn] pip install scikit-learn for classification report")

# =============================================================================
# SAVE MODEL + ARTIFACTS
# =============================================================================

WORK_DIR.mkdir(parents=True, exist_ok=True)

keras_path = WORK_DIR / "tomato_model.keras"
model.save(keras_path)
print(f"\nModel saved → {keras_path}")

with open(WORK_DIR / "tomato_class_names.json", "w", encoding="utf-8") as f:
    json.dump({
        "class_names": class_names,
        "crop": "tomato",
        "num_classes": num_classes,
        "dataset": DATASET_PATH,
    }, f, indent=2)

summary = {
    "class_names": class_names,
    "crop": "tomato",
    "img_size": IMG_SIZE,
    "normalization": {
        "scale": "divide_by_255",
        "mean": IMAGENET_MEAN.tolist(),
        "std": IMAGENET_STD.tolist(),
        "note": "Matches Agricai-Python ONNX inference",
    },
    "metrics": {
        "val_loss": float(loss),
        "val_accuracy": float(accuracy),
        "classification_report": classification_report_dict,
    },
    "target_accuracy": 0.98,
}
with open(WORK_DIR / "tomato_training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Export ONNX for Agricai-Python deployment
onnx_path = WORK_DIR / "tomato_model.onnx"
try:
    import onnx
    import tf2onnx

    spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, str(onnx_path))
    print(f"ONNX model saved → {onnx_path}")
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnx"])
    import onnx
    import tf2onnx
    spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, str(onnx_path))
    print(f"ONNX model saved → {onnx_path}")

# =============================================================================
# PLOT RESULTS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["accuracy"], label="Train")
axes[0].plot(history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["loss"], label="Train")
axes[1].plot(history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(WORK_DIR / "tomato_training_curves.png", dpi=120)
plt.show()

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Download from Kaggle Output:")
print("  - tomato_model.keras")
print("  - tomato_model.onnx          ← use in Agricai-Python")
print("  - tomato_class_names.json")
print("  - tomato_training_summary.json")
print("  - tomato_confusion_matrix.png")
print("  - tomato_training_curves.png")
if accuracy >= 0.98:
    print("\n✓ Reached 98% validation accuracy!")
else:
    print(f"\n→ Current: {accuracy * 100:.1f}% — add more images or train longer to reach 98%")
print("=" * 60)
