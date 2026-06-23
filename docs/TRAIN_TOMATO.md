# Train tomato leaf disease model

**Dataset layout:** See [DATASET_STRUCTURE.md](./DATASET_STRUCTURE.md) for the full folder standard (train/validation, naming, new crops, and tomato reference).

Scripts:
- **Disease model (10 or 11 classes):** [`training/train_tomato.py`](../training/train_tomato.py)
- **Stage 1 gate (tomato vs not):** [`training/train_tomato_gate.py`](../training/train_tomato_gate.py)
- **Kaggle notebook:** [`training/kaggle_train_tomato.py`](../training/kaggle_train_tomato.py)

## How it works (plain English)

1. **Stage 1 — Tomato leaf gate** (`tomato_leaf_gate.keras`)  
   A small binary model answers: *“Is this a tomato leaf?”*  
   If the answer is no (face, maize leaf, document, etc.), the API returns **“Not a tomato leaf”** and never runs the disease model.

2. **Stage 2 — Disease classifier** (`tomato_model.keras`)  
   Only runs when Stage 1 passes. Picks one of 10 tomato diseases/healthy.

3. **Optional 11th class — `Not_Tomato`**  
   If you add a `Not_Tomato/` folder to training data, the disease model can also learn to reject non-tomato images itself (backup to the gate).

## 1. Get the dataset

Use Kaggle **tomatoes-leaf-disease-detection** (PlantVillage-style folders).

**Minimum layout (10 tomato classes):**

```
tomatoes-leaf-disease-detection/
  train/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___healthy/
    ...
  validation/
    (same folder names)
```

**Recommended layout (gate + 11th reject class):**

```
tomatoes-leaf-disease-detection/
  train/
    Tomato___Early_blight/
    ...
    Tomato___healthy/
    Not_Tomato/          ← 500+ non-tomato images
      maize_leaves/
      faces/
      documents/
      random_photos/
  validation/
    Not_Tomato/
    (tomato folders)
```

`Not_Tomato/` should include: other crop leaves, indoor scenes, faces, screenshots, fruits, random objects — anything users might upload by mistake.

## 2. Train (Kaggle GPU or local)

```bash
cd Agricai-Python
pip install tensorflow scikit-learn matplotlib tf2onnx onnx onnxruntime

set TOMATO_DATASET_PATH=C:\path\to\tomatoes-leaf-disease-detection
set AGRICAI_WORK_DIR=.\model
python training/train_tomato.py
```

If `Not_Tomato/` exists, the script also trains the Stage 1 gate automatically.

**Gate only** (if you already have a disease model):

```bash
python training/train_tomato_gate.py
```

## 3. Deploy in AGRIC AI

Copy outputs to `model/`:

- `tomato_classifier.keras` → `model/tomato_model.keras`
- `tomato_class_names.json` → `model/tomato_class_names.json`
- `tomato_leaf_gate.keras` → `model/tomato_leaf_gate.keras`

Then:

```bash
python scripts/setup_tomato_model.py
uvicorn app.main:app --reload --port 8000
```

Important `.env` settings:

```env
INFERENCE_MODE=keras
MODEL_PATH=model/tomato_model.keras
CLASSES_PATH=data/classes_tomato.json
KERAS_PREPROCESS=imagenet
TOMATO_GATE_ENABLED=true
TOMATO_GATE_PATH=model/tomato_leaf_gate.keras
TOMATO_GATE_THRESHOLD=0.55
CONFIDENCE_THRESHOLD=0.58
```

## 4. Tips for high accuracy

- **100+ images per tomato disease class**; **500+ images in `Not_Tomato/`**
- Add **real phone photos** from the field, not only clean dataset images
- After training, tune thresholds: `python scripts/tune_thresholds.py field_test/`
- If real tomato leaves get rejected, lower `TOMATO_GATE_THRESHOLD` (e.g. 0.50)
- If wrong images still pass, raise it (e.g. 0.65) or add more `Not_Tomato` examples

## 5. What users see when an image is rejected

| Reason | Meaning |
|--------|---------|
| `not_tomato` | Gate or model says this is not a tomato leaf |
| `plant_guard` | Image does not look like any crop leaf (heuristic) |
| `low_confidence` | Model unsure — ask for a clearer photo |
| `low_margin` | Two diseases look equally likely |
