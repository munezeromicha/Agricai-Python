"""Bilingual copy when the API returns ``unknown`` (safety gates, not a disease label)."""

from __future__ import annotations

from app.inference.classify import RejectionReason

_MESSAGES: dict[RejectionReason | str, tuple[str, str, str, str]] = {
    "plant_guard": (
        "Not a crop leaf photo",
        "Si ifoto y'ikibabi cy'igihingwa",
        "This does not look like a crop leaf. Upload a clear, close-up photo of one leaf in daylight — not a screenshot or document.",
        "Ifoto ntiyirasa neza n'ikibabi. Shyiraho ifoto yegereye y'ikibabi kimwe mu mucyo mwiza.",
    ),
    "not_tomato": (
        "Not a tomato leaf",
        "Si ikibabi cy'inyanya",
        "This image does not look like a tomato leaf. Our model is trained for tomato leaves only — use a close-up photo of one tomato leaf, or choose the correct crop if we support it.",
        "Ifoto ntisa neza n'ikibabi cy'inyanya. Moderi yacu yigishijwe gusa ku mababi y'inyanya — shyiraho ifoto yegereye y'ikibabi kimwe cy'inyanya.",
    ),
    "low_confidence": (
        "Uncertain — try a clearer photo",
        "Ntibizwi — ongera ugerageze",
        "The model could not confirm a disease with enough confidence. Fill the frame with one leaf, hold steady, use even daylight, and show visible spots or discoloration if present.",
        "Moderi ntiyemeza indwara n'ukuri kuhagije. Fata ifoto y'ikibabi kimwe mu mucyo mwiza, igaragaza ibimenyetso niba bihari.",
    ),
    "low_margin": (
        "Similar conditions — need a sharper photo",
        "Indwara zisa — ongera ugerageze",
        "Two or more conditions look equally likely. Take a sharper close-up of the affected area only (one leaf, symptoms in focus).",
        "Indwara zirenze imwe zisa. Fata ifoto yegereye, isobanutse, yerekana gusa ahantu hari ibimenyetso.",
    ),
    "class_count_mismatch": (
        "Model configuration error",
        "Ikosa rya moderi",
        "The deployed model does not match the class list on the server. Contact support or redeploy the latest ONNX and classes.json.",
        "Moderi n'urutonde rw'indwara ntibihura. Vugana n'itsinda ryacu cyangwa ongera ushyiremo moderi nshya.",
    ),
}


def unknown_copy(reason: RejectionReason | None) -> tuple[str, str, str, str]:
    """Return diseaseName, diseaseNameRw, explanation, explanationRw."""
    key = reason or "low_confidence"
    return _MESSAGES.get(key, _MESSAGES["low_confidence"])
