# Reducing the generalization gap (train vs field photos)

## What we changed

| Layer | Change |
| --- | --- |
| **Inference** | Test-time augmentation (TTA): average logits over original, flip, and tight crops |
| **Gates** | `CONFIDENCE_THRESHOLD=0.58`, `CONFIDENCE_MARGIN=0.10` (was 0.80 / 0.18 in many `.env` files) |
| **Plant guard** | Allows brown/rust/diseased leaves, not only bright green |
| **API** | `rejection_reason`, `alternatives`, `top_confidence_pct` for clearer UX |
| **Training** | Stronger augmentation in `training/kaggle_train_full.py` (re-run Kaggle to benefit) |

## Debug a single image

From `Agricai-Python` with venv active:

```bash
python scripts/debug_predict.py path/to/leaf.jpg
python scripts/debug_predict.py path/to/folder --limit 10
python scripts/debug_predict.py leaf.jpg --no-tta
```

Look for:

- `plant_guard: BLOCKED` → photo failed color heuristics (not ONNX)
- `UNCERTAIN (low_confidence)` → top score below threshold
- `UNCERTAIN (low_margin)` → two classes too close (similar symptoms or wrong crop in frame)

## Tune thresholds on your own photos

1. Create a folder with **subfolders named exactly like `class_id`** in `data/classes.json`.
2. Put 10–30 field photos per class you care about.
3. Run:

```bash
python scripts/tune_thresholds.py field_test/
```

4. Copy suggested values into `.env` and restart the API (`pm2 restart Agricai-Python` or `uvicorn`).

## Retrain for best results

Current ONNX weights were trained with lighter augmentation. For a lasting fix:

1. Re-run `training/kaggle_train_full.py` on Kaggle (GPU).
2. Download `crop_classifier.onnx` and `class_names.json`.
3. `python scripts/sync_classes_json.py` (if class list changed).
4. Deploy ONNX + restart API.

Add **your uncertain field photos** into `train/` (correct class folders) before retraining — that directly closes the generalization gap.

## Photo guidelines (show in app)

- One leaf, close-up, fills most of the frame  
- Even daylight, in focus  
- Show symptoms (spots, rust, blight) when diseased  
- Match the crop (tomato model → tomato leaf)
