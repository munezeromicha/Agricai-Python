# Train tomato leaf disease model

Scripts:
- **Kaggle notebook (recommended):** [`training/kaggle_train_tomato.py`](../training/kaggle_train_tomato.py) — paste into one cell
- **Local / generic:** [`training/train_tomato.py`](../training/train_tomato.py)

## 1. Get the dataset

Download or use Kaggle dataset **tomatoes-leaf-disease-detection** (or PlantVillage tomato classes).

Folder layout:

```
tomatoes-leaf-disease-detection/
  train/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___Tomato_mosaic_virus/
    Tomato___Tomato_Yellow_Leaf_Curl_Virus/
    Tomato___healthy/
  validation/
    (same folder names)
```

## 2. Kaggle (recommended — free GPU)

1. Create a new notebook on [kaggle.com](https://www.kaggle.com)
2. **Settings → Accelerator → GPU T4**
3. **Add Data** → your tomato dataset
4. Upload `train_tomato.py` or paste its contents into a cell
5. Run:

```python
import os
os.environ["AGRICAI_WORK_DIR"] = "/kaggle/working/tomato_model"
%run train_tomato.py
```

6. Download from **Output**:
   - `tomato_classifier.onnx`
   - `tomato_class_names.json`
   - `tomato_training_summary.json`

## 3. Local training

```bash
cd Agricai-Python
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install tensorflow scikit-learn matplotlib tf2onnx onnx onnxruntime

set TOMATO_DATASET_PATH=C:\path\to\tomatoes-leaf-disease-detection
set AGRICAI_WORK_DIR=.\model\tomato
python training/train_tomato.py
```

## 4. Use the tomato model in AGRIC AI (testing — tomato only)

After training, place `tomato_model.keras` in `model/` (or `docs/`) and run:

```bash
python scripts/setup_tomato_model.py
```

This will:
- Copy `docs/tomato_model.keras` → `model/tomato_model.keras`
- Generate `data/classes_tomato.json` (10 tomato classes + unknown)
- Set `.env` to `INFERENCE_MODE=keras` (old multi-crop ONNX moved to `model/archive/`)

```env
INFERENCE_MODE=keras
MODEL_PATH=model/tomato_model.keras
CLASSES_PATH=data/classes_tomato.json
MODEL_VERSION=tomato-cnn-1.0.0
KERAS_PREPROCESS=builtin_rescale
```

Restart: `uvicorn app.main:app --reload --port 8000`

To restore the full multi-crop ONNX model later, set `INFERENCE_MODE=onnx`, `MODEL_PATH=model/archive/crop_classifier.onnx`, and `CLASSES_PATH=data/classes_multicrop.json.bak` (rename back to `classes.json`).

## 5. Tips for 98%+ accuracy

- At least **100+ images per class** (more for rare diseases)
- Include **phone photos** from Rwanda fields, not only dataset images
- Balance classes (script uses `class_weight` automatically)
- After training, run `python scripts/tune_thresholds.py` on `field_test/` images
