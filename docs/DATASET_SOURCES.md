# AGRIC AI — Where to get training datasets (by crop)

This guide maps **every crop currently in your knowledge base** to public datasets you can download, then prepare with `scripts/prepare_crop_dataset.py` into the layout required by [DATASET_STRUCTURE.md](./DATASET_STRUCTURE.md).

**Rule for your client problem:** each crop model must include a `Not_<Crop>/` reject class built from **other crop leaves** (hard negatives). Soft negatives alone (faces, documents) will not stop wrong-plant detections.

---

## 1. Crops you already have labeled data for

From `data/classes.json` + tomato pipeline:

| Crop | Classes in project today | Best public source | Ready quality |
|------|--------------------------|--------------------|---------------|
| **Tomato** | 10 diseases + healthy (+ `Not_Tomato`) | PlantVillage tomato / tomatoes-leaf-disease-detection | Excellent |
| **Maize** | Blight, Common rust, Gray leaf spot, Healthy | PlantVillage corn + Africa maize sets | Excellent |
| **Potato** | Early blight, Late blight, Healthy | PlantVillage potato | Excellent |
| **Beans** | Angular leaf spot, Rust, Healthy | CIAT / Makerere beans (often on Kaggle) | Good |
| **Onion** | Several leaf + bulb classes | TOM2024 onion (Mendeley) | Good (field photos) |
| **Mango** | Anthracnose, Gall midge, Powdery mildew, Healthy | MangoLeafBD (Kaggle) | Good |
| **Orange** | Canker, Fresh, Greening | Orange diseases dataset (Kaggle) — **fruit**, not leaf | Medium (reframe carefully) |
| **Tea** | Algal leaf, bird eye spot, brown blight, gray blight, healthy, red/white spot, anthracnose | Tea sickness / teaLeafBD (Kaggle) | Good |

Marketing crops in Node (`avocado`, `carrots`, …) with **empty disease lists** are **not** ready for training yet — collect/label data before promising detection.

---

## 2. Recommended download list (start here)

Add these as **Kaggle Input** datasets (or download ZIP → unzip locally).

### Priority A — train first (client-facing reliability)

| # | Crop | Dataset | Link / search term |
|---|------|---------|--------------------|
| 1 | Tomato | Tomatoes leaf disease (PlantVillage-style) | Kaggle: `tomatoes-leaf-disease-detection` or PlantVillage tomato folders |
| 2 | Maize | PlantVillage corn + Africa maize | Kaggle: plantvillage corn / `maize-beans-and-tomatoes-image-dataset-for-africa` |
| 3 | Potato | PlantVillage potato | Included in PlantVillage mirrors |
| 4 | Beans | Bean leaf disease (Uganda/Tanzania style) | Kaggle: bean disease / Africa maize-beans-tomatoes pack |

### Priority B — expand after tomato works

| # | Crop | Dataset | Link / search term |
|---|------|---------|--------------------|
| 5 | Mango | MangoLeafBD | https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset |
| 6 | Tea | Tea sickness / teaLeafBD | https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs · https://www.kaggle.com/datasets/bmshahriaalam/tealeafbd-tea-leaf-disease-detection |
| 7 | Onion + Maize + Tomato (field) | **TOM2024** | https://data.mendeley.com/datasets/3d4yg89rtr/1 |
| 8 | Orange | Orange diseases | https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset |

### Africa-focused multi-crop pack (optional)

- **Maize, Beans and Tomatoes for Africa** (~68k images, 32 classes):  
  https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa  

Use this to **add field diversity** after you have clean PlantVillage-style folders — do not mix naming blindly.

---

## 3. Folder name mapping (source → AGRIC AI)

After download, rename class folders to **exactly** these names (or use the prepare script).

### Tomato (`Not_Tomato` required)

| Source folder (typical) | Use as |
|-------------------------|--------|
| `Tomato___Bacterial_spot` | `Tomato___Bacterial_spot` |
| `Tomato___Early_blight` | `Tomato___Early_blight` |
| `Tomato___Late_blight` | `Tomato___Late_blight` |
| `Tomato___Leaf_Mold` | `Tomato___Leaf_Mold` |
| `Tomato___Septoria_leaf_spot` | `Tomato___Septoria_leaf_spot` |
| `Tomato___Spider_mites Two-spotted_spider_mite` | keep exact name |
| `Tomato___Target_Spot` | `Tomato___Target_Spot` |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | same |
| `Tomato___Tomato_mosaic_virus` | same |
| `Tomato___healthy` | `Tomato___healthy` |

### Maize

| Source | Use as |
|--------|--------|
| Corn / Maize Northern Leaf Blight | `Maize___Northern_Leaf_Blight` or keep `Maize_Blight` if matching old JSON |
| Common Rust | `Maize___Common_Rust` |
| Gray Leaf Spot | `Maize___Gray_Leaf_Spot` |
| Healthy | `Maize___Healthy` |

Prefer the `Crop___Condition` style going forward; then update `data/classes_maize.json` to match.

### Potato

| Source | Use as |
|--------|--------|
| `Potato___Early_blight` | same |
| `Potato___Late_blight` | same |
| `Potato___healthy` | same |

### Beans

| Source | Use as |
|--------|--------|
| Angular Leaf Spot | `Bean___Angular_leaf_spot` |
| Bean Rust | `Bean___Bean_rust` |
| Healthy | `Bean___healthy` |

### Mango (MangoLeafBD)

| Source | Use as |
|--------|--------|
| Anthracnose | `Mango___Anthracnose` |
| Gall Midge | `Mango___Gall_Midge` |
| Powdery Mildew | `Mango___Powdery_Mildew` |
| Healthy | `Mango___Healthy` |
| (optional) Bacterial Canker, Die Back, … | add only if you also add JSON advice text |

### Tea

| Source (tea sickness) | Use as |
|-----------------------|--------|
| algal leaf | `Tea___Algal_leaf` |
| bird eye spot | `Tea___Bird_eye_spot` |
| brown blight | `Tea___Brown_blight` |
| gray light / gray blight | `Tea___Gray_blight` |
| red leaf spot | `Tea___Red_leaf_spot` |
| white spot | `Tea___White_spot` |
| Anthracnose | `Tea___Anthracnose` |
| healthy | `Tea___Healthy` |

### Onion (TOM2024)

Map expert labels carefully (Caterpillar, Fusarium, Purple blotch, Downy mildew, Healthy leaf, Bulb rot, …) → `Onion___…` folders. Prefer **leaf** classes for the phone scanner; bulb rot is a different organ — keep separate or exclude from leaf model.

### Orange

This public set is mostly **fruit** (canker / greening / fresh). For leaf scanning in the app, either:

- train a **fruit** orange model and label the UI accordingly, or  
- wait until you have orange **leaf** photos from the field.

---

## 4. How to prepare one crop (step by step)

### Step 1 — Download

1. Create a Kaggle account and verify phone (needed for large datasets / GPU).  
2. Download ZIP for Priority A crop #1 (Tomato).  
3. Unzip to a working folder, e.g. `D:\datasets\raw\tomato\`.

### Step 2 — Run prepare script

From `Agricai-Python` (venv active):

```powershell
python scripts/prepare_crop_dataset.py `
  --crop Tomato `
  --source D:\datasets\raw\tomato `
  --out D:\datasets\prepared\tomato `
  --negatives D:\datasets\raw\maize D:\datasets\raw\beans D:\datasets\raw\potato `
  --val-fraction 0.15 `
  --test-fraction 0.15
```

What it does:

- Discovers class folders under `source`  
- Renames via optional `--map` JSON (or keeps PlantVillage-style names)  
- Splits into `train/` / `validation/` / `test/`  
- Builds `Not_Tomato/` from images in `--negatives` (other crops)

### Step 3 — Knowledge base

Create/update `data/classes_tomato.json` so every **folder name** = `class_id`, with:

- disease name (EN + RW)  
- explanation / cause  
- treatment  
- prevention  
- care  

(This text is what your client sees after detection — not the CNN.)

### Step 4 — Train on Kaggle

Upload the **prepared** folder as a Kaggle dataset, attach it to a notebook, run `training/kaggle_train_tomato.py` (or crop-specific script) with GPU on.

### Step 5 — Field negatives

Before client demo, add 100–300 **real phone** photos into each disease folder and into `Not_<Crop>/` (wrong crops, distant plants, soil). Retrain once — this is what stops “wrong leaf → fake disease.”

---

## 5. Suggested prepared tree (example: tomato)

```
prepared/tomato/
├── train/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── ...
│   ├── Tomato___healthy/
│   └── Not_Tomato/          ← maize + bean + potato leaves + junk
├── validation/
│   └── (same folders)
└── test/
    └── (same folders — never used for early stopping)
```

---

## 6. What NOT to do

| Mistake | Why it hurts |
|---------|----------------|
| Train all 8 crops in one model first | Confuses similar leaves; harder gates |
| `Not_*` = only faces/screenshots | Gate gets 100% val, fails on maize leaves |
| Use orange fruit dataset as “leaf disease” | App UX is leaf-first |
| Trust 98% when reject class is 79% of val | Report **macro F1** instead |
| Promise avocado/carrot detection with empty disease lists | No labeled data yet |

---

## 7. Practical order for Agric AI

1. **Tomato prepared + hard `Not_Tomato`** → train → deploy → client demo  
2. Maize prepared + `Not_Maize`  
3. Potato + Beans  
4. Mango + Tea  
5. Onion (TOM2024) with careful label mapping  
6. Orange only if you accept fruit photos or collect leaf data  

---

## 8. Related docs

- [DATASET_STRUCTURE.md](./DATASET_STRUCTURE.md) — folder rules  
- [TRAIN_TOMATO.md](./TRAIN_TOMATO.md) — tomato train / gate / deploy  
- `scripts/prepare_crop_dataset.py` — build prepared trees  
- `scripts/evaluate_field_test.py` / `tune_thresholds.py` — after training  
