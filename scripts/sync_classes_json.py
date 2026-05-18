"""
Build data/classes.json from model/class_names.json (same order as ONNX output indices).
Run from Agricai-Python root: python scripts/sync_classes_json.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_PATH = ROOT / "model" / "class_names.json"
OUT_PATH = ROOT / "data" / "classes.json"

GENERIC_EXPLANATION_EN = (
    "Our vision model matched this image to a known condition in the training set. "
    "Confirm with a local agronomist before applying chemicals."
)
GENERIC_EXPLANATION_RW = (
    "Moderi yacu yahuye ifoto n'ikimenyetso cy'imenyekana mu masomo. "
    "Emeza n'umuhinzi w'impuguke mbere yo gukoresha imiti."
)
GENERIC_TREATMENT_EN = (
    "Follow label directions for crop-approved products. Remove heavily affected leaves. "
    "Do not mix chemicals unless advised by extension services."
)
GENERIC_TREATMENT_RW = (
    "Kurikiza ibisobanuro by'imiti yemewe ku bihingwa. Kuraho amababi menshi yangiritse. "
    "Ntuhuze imiti utabifitiye inama."
)
GENERIC_PREVENTION_EN = (
    "Use clean planting material, rotate crops, avoid overhead irrigation, and scout fields weekly."
)
GENERIC_PREVENTION_RW = (
    "Koresha imbuto/isigo nbyo byiza, hindura ibihingwa, wirinde kuhira hejuru, usuzume umurima buri cyumweru."
)
GENERIC_CARE_EN = "Monitor spread after rain; keep good airflow; record symptoms and dates."
GENERIC_CARE_RW = "Kurikirana nyuma y'imvura; komeza umuyaga; andika ibimenyetso n'itariki."
HEALTHY_TREATMENT_EN = "No treatment needed. Continue good field practices and routine scouting."
HEALTHY_TREATMENT_RW = "Nta muti ukenewe. Komeza imikorere myiza yo mu murima."


def humanize(class_id: str) -> str:
    s = class_id.replace("___", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def infer_type(class_id: str) -> str:
    low = class_id.lower()
    if "healthy" in low or "fresh" in low and "orange" in low:
        return "healthy"
    if any(x in low for x in ("caterpillar", "midge", "virosis", "virus", "pest")):
        return "pest"
    return "disease"


def entry_for(class_id: str) -> dict:
    display = humanize(class_id)
    kind = infer_type(class_id)
    return {
        "class_id": class_id,
        "type": kind,
        "diseaseName": display,
        "diseaseNameRw": display,
        "explanation": GENERIC_EXPLANATION_EN,
        "explanationRw": GENERIC_EXPLANATION_RW,
        "treatment": HEALTHY_TREATMENT_EN if kind == "healthy" else GENERIC_TREATMENT_EN,
        "treatmentRw": HEALTHY_TREATMENT_RW if kind == "healthy" else GENERIC_TREATMENT_RW,
        "prevention": GENERIC_PREVENTION_EN,
        "preventionRw": GENERIC_PREVENTION_RW,
        "care": GENERIC_CARE_EN,
        "careRw": GENERIC_CARE_RW,
    }


def main() -> None:
    raw = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    names: list[str] = raw["class_names"] if isinstance(raw, dict) else raw

    classes = [entry_for(name) for name in names]
    classes.append(
        {
            "class_id": "unknown",
            "type": "unknown",
            "diseaseName": "Uncertain diagnosis",
            "diseaseNameRw": "Ntibizwi neza",
            "explanation": (
                "The image could not be matched with high confidence. "
                "Try a clearer photo of the affected leaf in natural light."
            ),
            "explanationRw": (
                "Ifoto ntiyahuje n'ukuri guhanitse. Gerageza ifoto yizewe y'ikibabi mu mucyo wa kare."
            ),
            "treatment": "Seek local extension advice before applying chemicals.",
            "treatmentRw": "Shakisha inama y'impuguke mbere yo gukoresha imiti.",
            "prevention": GENERIC_PREVENTION_EN,
            "preventionRw": GENERIC_PREVENTION_RW,
            "care": GENERIC_CARE_EN,
            "careRw": GENERIC_CARE_RW,
        }
    )

    OUT_PATH.write_text(
        json.dumps({"classes": classes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(classes)} rows ({len(names)} trainable + unknown) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
