# Deploy tomato models (Kaggle zip → production)

GitHub stores **code only**. Model weights stay on your machine or server (`.keras` files are ~24 MB each; the full Kaggle zip is ~371 MB and must not be committed).

## What `model/agricai_models.zip` contains

Kaggle training writes one zip with everything. For inference you only need **6 files**:

| File inside zip | Deployed as |
| --- | --- |
| `tomato_classifier.keras` | `model/tomato_model.keras` |
| `tomato_class_names.json` | `model/tomato_class_names.json` |
| `tomato_leaf_gate.keras` | `model/tomato_leaf_gate.keras` |
| `tomato_gate_summary.json` | `model/tomato_gate_summary.json` |
| `tomato_training_summary.json` | `model/tomato_training_summary.json` |

The zip also includes `gate_split_data/` (thousands of temp images). Those are **not** needed at runtime — `scripts/deploy_kaggle_models.py` skips them.

## Local or server setup

1. Clone the repo and install dependencies:

```bash
cd Agricai-Python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

2. Place models on the server using **one** of these options:

**Option A — zip (recommended after Kaggle)**

Copy `agricai_models.zip` to `model/` (SCP, USB, cloud storage — not git), then:

```bash
python scripts/deploy_kaggle_models.py
```

**Option B — extracted files**

Copy these into `model/` manually, then:

```bash
python scripts/setup_tomato_model.py
```

3. Confirm `.env` (production example):

```env
INFERENCE_MODE=keras
MODEL_PATH=model/tomato_model.keras
CLASSES_PATH=data/classes_tomato.json
MODEL_VERSION=tomato-cnn-2.0.0
KERAS_PREPROCESS=imagenet
TOMATO_GATE_ENABLED=true
TOMATO_GATE_PATH=model/tomato_leaf_gate.keras
TOMATO_GATE_THRESHOLD=0.52
TOMATO_GATE_HARD_REJECT=0.38
CONFIDENCE_THRESHOLD=0.62
CONFIDENCE_MARGIN=0.10
NOT_TOMATO_COMPETE_THRESHOLD=0.32
NOT_TOMATO_COMPETE_MARGIN=0.15
PLANT_GUARD_MIN_SCORE=0.42
CORS_ORIGINS=https://your-frontend-domain.com
```

4. Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Production (Linux + PM2): `./scripts/pm2-deploy.sh`

5. Point the frontend at this service:

```env
VITE_VISION_API_URL=https://ml.your-domain.com
```

## Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
python scripts/debug_predict.py path/to/tomato_leaf.jpg
```

## Sharing models without git

| Method | When to use |
| --- | --- |
| **SCP / SFTP** | Single VPS deploy |
| **Google Drive / S3 / Kaggle dataset** | Team download link |
| **GitHub Release asset** | Attach zip to a release (not the repo tree) |
| **Git LFS** | Only if you want models versioned in git (paid bandwidth on GitHub) |

Keep `model/tomato_class_names.json` and `model/*_summary.json` in git — they are small metadata. Keep `.keras` and `.zip` out of git.
