# Agric AI — Vision API

FastAPI service for crop image diagnosis. Ships with a **stub** engine (no weights) and optional **ONNX** inference for your trained classifier.

## Setup

```bash
cd Agricai-Python
python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Linux/macOS: cp .env.example .env
```

## Run (development)

From the **project root** (the folder that contains `app/`):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m app
```

| Uvicorn target | Meaning |
| --- | --- |
| `app.main` | Python module `app/main.py` |
| `:app` | FastAPI instance named `app` in that file |

Do **not** use `main:app` — there is no `main.py` at the repository root.

- Root: `GET http://localhost:8000/`
- Health: `GET http://localhost:8000/health`
- Detect: `POST http://localhost:8000/v1/detect` — form field `file` (image)
- Model metadata: `GET http://localhost:8000/v1/models`

## Production (PM2 on Linux)

1. Create the venv and install dependencies on the server (paths below assume repo root `/root/backend/Agricai/Agricai-Python`).

2. Copy `.env` and set `INFERENCE_MODE`, `MODEL_PATH`, `CORS_ORIGINS`, etc.

3. Remove any broken PM2 process, then start with the checked-in config:

```bash
cd /root/backend/Agricai/Agricai-Python
pm2 delete Agricai-Python 2>/dev/null || true
pm2 start ecosystem.config.cjs
pm2 save
```

`ecosystem.config.cjs` runs `.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000` with `cwd` set to the repo root.

Manual equivalent (note **`app.main:app`**, not `main:app`):

```bash
pm2 start .venv/bin/uvicorn \
  --name Agricai-Python \
  --cwd /root/backend/Agricai/Agricai-Python \
  -- app.main:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

Shell helper (no PM2): `chmod +x scripts/start-production.sh && ./scripts/start-production.sh`

Windows (no PM2): `.\scripts\start-production.ps1`

Put **Caddy** (or another reverse proxy) in front of port `8000` for `ml.agric-ai.com`.

## ONNX model

1. Export a classifier that outputs **logits** with shape `[1, N]` where **N equals** the number of entries in `data/classes.json` (class index `i` must match row `i` in that file).
2. Set `INFERENCE_MODE=onnx`, `MODEL_PATH=...`, and `MODEL_VERSION=...` in `.env`.
3. Preprocessing matches common ImageNet settings: resize to `INPUT_SIZE`, RGB, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, NCHW float32. Change `app/inference/engine.py` if your training recipe differs.

## Knowledge base

Bilingual copy (EN / Kinyarwanda) and UI fields are edited in `data/classes.json`, keyed by `class_id`. The vision model only selects which row to show; it does not generate pesticide text.

## Contact form (Node + nodemailer)

The **Get in Touch** form on the marketing site posts to a small Express API that sends mail via SMTP.

```bash
cd contact-api
npm install
copy .env.example .env
npm run dev
```

- Health: `GET http://localhost:3008/health`
- Submit: `POST http://localhost:3008/api/contact` with JSON `{ "name", "email", "subject", "message" }`

Set `MAIL_TO` to the inbox that should receive submissions, and configure `SMTP_*` in `.env`. In the Vite app, set `VITE_CONTACT_API_URL=http://localhost:3008` (see `contact-api/.env.example`).
