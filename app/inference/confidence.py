"""Confidence banding for detection results.

Mirrors `Agricai-Node/src/lib/confidence.mjs` exactly: the vision API, the platform
API and the app must all describe the same number with the same word, otherwise a
farmer sees "high confidence" on one screen and "check again" on the next.

Rules: the score picks the band, and a thin margin to the runner-up class demotes it
by exactly one step (two diseases that look alike deserve a second photo, not a
discarded result).
"""

from __future__ import annotations

from typing import Literal

ConfidenceLevel = Literal["high", "medium", "low", "very_low"]

# (minimum score, minimum margin to the runner-up) per band, in percentage points.
BANDS: dict[str, tuple[float, float]] = {
    "high": (85.0, 12.0),
    "medium": (65.0, 6.0),
    "low": (45.0, 0.0),
}
_ORDER: list[ConfidenceLevel] = ["high", "medium", "low", "very_low"]


def confidence_level(confidence_pct: float | None, margin_pct: float | None = None) -> ConfidenceLevel:
    """Return the band for a top-1 confidence (0–100) and optional runner-up margin."""
    try:
        conf = float(confidence_pct) if confidence_pct is not None else 0.0
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(100.0, conf))

    margin: float | None
    try:
        margin = float(margin_pct) if margin_pct is not None else None
    except (TypeError, ValueError):
        margin = None

    band: ConfidenceLevel
    if conf >= BANDS["high"][0]:
        band = "high"
    elif conf >= BANDS["medium"][0]:
        band = "medium"
    elif conf >= BANDS["low"][0]:
        band = "low"
    else:
        return "very_low"

    required_margin = BANDS[band][1]
    if margin is not None and margin < required_margin:
        return _ORDER[min(_ORDER.index(band) + 1, len(_ORDER) - 1)]
    return band


def confidence_guidance(level: ConfidenceLevel) -> tuple[str, str]:
    """Plain-language (English, Kinyarwanda) guidance for a band."""
    if level == "high":
        return (
            "High confidence — the model matched this leaf strongly. You can act on the treatment advice.",
            "Ukuri kwinshi — moderi yamenye neza iki kibabi. Ushobora gukurikiza inama zo kuvura.",
        )
    if level == "medium":
        return (
            "Medium confidence — likely correct, but take a second photo of another affected leaf before spraying.",
            "Ukuri kuringaniye — birashoboka ko ari byo, ariko fata indi foto y'ikindi kibabi mbere yo gufumbira.",
        )
    if level == "low":
        return (
            "Low confidence — treat this as a hint only. Re-scan in better light or ask an agronomist.",
            "Ukuri guke — bifate nk'icyerekezo gusa. Ongera usuzume mu mucyo mwiza cyangwa ubaze umujyanama.",
        )
    return (
        "Very low confidence — do not act on this result. Retake the photo with one leaf filling the frame.",
        "Ukuri guke cyane — ntukurikize iki gisubizo. Ongera ufate ifoto y'ikibabi kimwe cyuzuye.",
    )


def is_actionable(level: ConfidenceLevel) -> bool:
    """True when the result is solid enough to justify buying and applying a treatment."""
    return level in ("high", "medium")
