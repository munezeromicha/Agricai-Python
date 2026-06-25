"""Roboflow serverless object-detection inference (tomato disease lesions)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.inference.knowledge import KnowledgeBase
from app.schemas import ClassAlternative, DetectResponse, DetectionBox, DetectionResult

_CLASS_COLORS: dict[str, str] = {
    "early blight": "#14b8a6",
    "late blight": "#ec4899",
    "bacterial spot": "#22c55e",
    "septoria": "#f97316",
    "septoria leaf spot": "#f97316",
    "leaf mold": "#06b6d4",
    "spider mites": "#84cc16",
    "target spot": "#eab308",
    "yellow leaf curl virus": "#a855f7",
    "mosaic virus": "#8b5cf6",
    "healthy": "#22c55e",
}
_FALLBACK_COLORS = ["#14b8a6", "#ec4899", "#f97316", "#22c55e", "#8b5cf6", "#06b6d4"]


def _normalize_label(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _color_for_class(class_name: str) -> str:
    key = _normalize_label(class_name)
    if key in _CLASS_COLORS:
        return _CLASS_COLORS[key]
    for partial, color in _CLASS_COLORS.items():
        if partial in key or key in partial:
            return color
    idx = sum(ord(c) for c in key) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def _map_roboflow_class_to_kb(class_name: str, kb: KnowledgeBase) -> str | None:
    norm = _normalize_label(class_name)
    compact = norm.replace(" ", "_")
    candidates = [
        f"Tomato___{compact}",
        f"Tomato___{'_'.join(w.capitalize() for w in norm.split())}",
    ]
    if norm.startswith("septoria"):
        candidates.insert(0, "Tomato___Septoria_leaf_spot")
    if "early blight" in norm:
        candidates.insert(0, "Tomato___Early_blight")
    if "late blight" in norm:
        candidates.insert(0, "Tomato___Late_blight")
    if "bacterial" in norm:
        candidates.insert(0, "Tomato___Bacterial_spot")

    for cid in candidates:
        if kb.try_get(cid) is not None:
            return cid
    for cid in kb.trainable_class_ids:
        entry = kb.get(cid)
        dn = _normalize_label(entry.diseaseName)
        if dn == norm or norm in dn or dn in norm:
            return cid
    return None


def _prepare_image_bytes(image_bytes: bytes, max_edge: int = 1280) -> tuple[bytes, int, int]:
    pil = Image.open(BytesIO(image_bytes))
    pil.load()
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    w, h = pil.size
    if max(w, h) > max_edge:
        pil.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        w, h = pil.size
        out = BytesIO()
        pil.save(out, format="JPEG", quality=92)
        return out.getvalue(), w, h
    return image_bytes, w, h


def _multipart_body(
    boundary: bytes,
    *,
    file_name: str,
    file_bytes: bytes,
    mime: str,
) -> bytes:
    crlf = b"\r\n"
    parts: list[bytes] = [
        b"--" + boundary + crlf,
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"'.encode() + crlf,
        f"Content-Type: {mime}".encode() + crlf + crlf,
        file_bytes + crlf,
        b"--" + boundary + b"--" + crlf,
    ]
    return b"".join(parts)


def _call_roboflow_api(image_bytes: bytes, settings: Settings) -> tuple[dict[str, Any], int, int]:
    """POST raw image bytes to Roboflow serverless (same as their web UI)."""
    image_bytes, img_w, img_h = _prepare_image_bytes(image_bytes)

    model_id = settings.roboflow_model_id.strip("/")
    url = f"{settings.roboflow_api_url.rstrip('/')}/{model_id}"

    # Roboflow API filters server-side; their UI fetches low then filters in browser.
    # Use a low API threshold so we get boxes, then filter in the portal slider.
    api_conf_pct = max(1, min(40, settings.roboflow_api_confidence_pct))
    overlap_pct = int(round(settings.roboflow_iou_threshold * 100)) if settings.roboflow_iou_threshold <= 1 else int(settings.roboflow_iou_threshold)

    query = urllib.parse.urlencode(
        {
            "api_key": settings.roboflow_api_key,
            "confidence": api_conf_pct,
            "overlap": overlap_pct,
        }
    )
    full_url = f"{url}?{query}"
    boundary = uuid.uuid4().hex.encode()
    payload = b""
    status = 0

    for attempt in range(2):
        try:
            body = _multipart_body(
                boundary,
                file_name="image.jpg",
                file_bytes=image_bytes,
                mime="image/jpeg",
            )
            request = urllib.request.Request(
                full_url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Roboflow API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == 0:
                image_bytes, img_w, img_h = _prepare_image_bytes(image_bytes, max_edge=960)
                boundary = uuid.uuid4().hex.encode()
                continue
            raise RuntimeError(f"Roboflow API connection failed: {exc}") from exc

    if status >= 400:
        detail = payload.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Roboflow API error {status}: {detail}")

    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise RuntimeError("Unexpected Roboflow response format.")

    img_meta = raw.get("image") or {}
    resp_w = int(img_meta.get("width") or img_w)
    resp_h = int(img_meta.get("height") or img_h)
    return raw, resp_w, resp_h


def _parse_predictions(raw: dict[str, Any]) -> list[dict[str, Any]]:
    preds = raw.get("predictions")
    if not isinstance(preds, list):
        preds = []
    return [p for p in preds if isinstance(p, dict)]


def _boxes_from_predictions(preds: list[dict[str, Any]]) -> list[DetectionBox]:
    return [
        DetectionBox(
            class_name=str(p.get("class") or "unknown"),
            class_id=str(p.get("class_id")) if p.get("class_id") is not None else None,
            confidence=round(float(p.get("confidence", 0)) * 100.0, 1),
            x=float(p.get("x", 0)),
            y=float(p.get("y", 0)),
            width=float(p.get("width", 0)),
            height=float(p.get("height", 0)),
            color=_color_for_class(str(p.get("class") or "unknown")),
        )
        for p in preds
    ]


def _primary_class(preds: list[dict[str, Any]]) -> tuple[str, float]:
    if not preds:
        return "unknown", 0.0
    best = max(preds, key=lambda p: float(p.get("confidence", 0)))
    return str(best.get("class", "unknown")), float(best.get("confidence", 0)) * 100.0


def run_roboflow_detect(
    image_bytes: bytes,
    *,
    settings: Settings | None = None,
    kb: KnowledgeBase | None = None,
) -> DetectResponse:
    settings = settings or get_settings()
    if not settings.roboflow_api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set in the environment.")

    kb = kb or KnowledgeBase(settings.resolved_classes_path)
    raw, img_w, img_h = _call_roboflow_api(image_bytes, settings)
    preds = _parse_predictions(raw)
    boxes = _boxes_from_predictions(preds)

    if not preds:
        unk = kb.get("unknown")
        return DetectResponse(
            result=DetectionResult(
                diseaseName="No disease regions detected",
                diseaseNameRw="Nta bice by'indwara byabonetse",
                confidence=0.0,
                type="unknown",
                explanation="No tomato disease lesions were found above the confidence threshold. Try a closer photo with visible spots.",
                explanationRw="Nta bice by'indwara byabonetse. Fata ifoto yegereye igaragaza ibimenyetso.",
                treatment=unk.treatment,
                treatmentRw=unk.treatmentRw,
                prevention=unk.prevention,
                preventionRw=unk.preventionRw,
                care=unk.care,
                careRw=unk.careRw,
            ),
            model_version=settings.model_version,
            request_id=str(uuid.uuid4()),
            inference_mode="roboflow",
            top_class_id="unknown",
            rejection_reason="low_confidence",
            alternatives=[],
            top_confidence_pct=0.0,
            detections=boxes,
            image_width=img_w,
            image_height=img_h,
            roboflow_model_id=settings.roboflow_model_id,
        )

    primary_name, primary_conf = _primary_class(preds)
    kb_id = _map_roboflow_class_to_kb(primary_name, kb)

    alternatives: list[ClassAlternative] = []
    seen: set[str] = set()
    for p in sorted(preds, key=lambda x: float(x.get("confidence", 0)), reverse=True):
        cname = str(p.get("class", ""))
        if cname in seen:
            continue
        seen.add(cname)
        cid = _map_roboflow_class_to_kb(cname, kb) or cname
        alternatives.append(
            ClassAlternative(
                class_id=cid,
                disease_name=cname,
                confidence=round(float(p.get("confidence", 0)) * 100.0, 1),
            )
        )

    if kb_id:
        entry = kb.get(kb_id)
        result = kb.to_detection(entry, primary_conf)
    else:
        unk = kb.get("unknown")
        result = DetectionResult(
            diseaseName=primary_name,
            diseaseNameRw=primary_name,
            confidence=round(primary_conf, 1),
            type="disease",
            explanation=f"Detected {primary_name} on {len(preds)} region(s). Confirm with a local agronomist.",
            explanationRw=f"Byabonetse {primary_name} mu bice {len(preds)}. Emeza n'umuhinzi w'impuguke.",
            treatment=unk.treatment,
            treatmentRw=unk.treatmentRw,
            prevention=unk.prevention,
            preventionRw=unk.preventionRw,
            care=unk.care,
            careRw=unk.careRw,
        )
        kb_id = primary_name

    result.type = "disease"  # type: ignore[misc]

    return DetectResponse(
        result=result,
        model_version=settings.model_version,
        request_id=str(uuid.uuid4()),
        inference_mode="roboflow",
        top_class_id=kb_id,
        rejection_reason=None,
        alternatives=alternatives[:12],
        top_confidence_pct=round(primary_conf, 1),
        detections=boxes,
        image_width=img_w,
        image_height=img_h,
        roboflow_model_id=settings.roboflow_model_id,
    )
