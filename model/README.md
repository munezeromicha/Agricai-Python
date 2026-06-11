# Model files (not in git)

Place trained weights here after cloning. Required for tomato inference:

- `tomato_model.keras` — 11-class disease classifier
- `tomato_leaf_gate.keras` — binary tomato-leaf gate
- `tomato_class_names.json` — class order (small; tracked in git)

Quick setup from Kaggle zip:

```bash
# Copy agricai_models.zip into this folder, then:
python scripts/deploy_kaggle_models.py
```

See [docs/DEPLOY_MODELS.md](../docs/DEPLOY_MODELS.md).
