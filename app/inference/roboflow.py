"""Roboflow serverless inference for multi-crop disease detection."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.inference.crops.registry import get_crop, resolve_workflow_id
from app.inference.crops.types import CropConfig
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
    "rust": "#eab308",
    "blight": "#f97316",
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


def _map_class_to_kb(class_name: str, kb: KnowledgeBase, crop: CropConfig) -> str | None:
    norm = _normalize_label(class_name)
    compact = norm.replace(" ", "_")
    compact_title = "_".join(w.capitalize() for w in norm.split())

    candidates: list[str] = []
    if crop.kb_prefix:
        candidates.extend(
            [
                f"{crop.kb_prefix}___{compact}",
                f"{crop.kb_prefix}_{compact}",
                f"{crop.kb_prefix}___{compact_title}",
                f"{crop.kb_prefix}_{compact_title}",
            ]
        )
    candidates.extend([compact, compact_title, class_name.strip()])

    if crop.id == "tomato":
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
        cid_norm = _normalize_label(cid.replace("___", " ").replace("_", " "))
        if cid_norm == norm or norm in cid_norm or cid_norm in norm:
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
    if pil.mode == "L":
        out = BytesIO()
        pil.convert("RGB").save(out, format="JPEG", quality=92)
        return out.getvalue(), w, h
    return image_bytes, w, h


def _multipart_body(
    boundary: bytes,
    *,
    file_name: str,
    file_bytes: bytes,
    mime: str,
    extra_fields: dict[str, str] | None = None,
) -> bytes:
    crlf = b"\r\n"
    parts: list[bytes] = []
    for name, value in (extra_fields or {}).items():
        parts.extend(
            [
                b"--" + boundary + crlf,
                f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf,
                value.encode("utf-8") + crlf,
            ]
        )
    parts.extend(
        [
            b"--" + boundary + crlf,
            f'Content-Disposition: form-data; name="file"; filename="{file_name}"'.encode() + crlf,
            f"Content-Type: {mime}".encode() + crlf + crlf,
            file_bytes + crlf,
            b"--" + boundary + b"--" + crlf,
        ]
    )
    return b"".join(parts)


def _http_post_multipart(
    url: str,
    *,
    image_bytes: bytes,
    api_key: str,
    query: dict[str, str | int] | None = None,
    extra_fields: dict[str, str] | None = None,
) -> bytes:
    params = {"api_key": api_key, **(query or {})}
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    boundary = uuid.uuid4().hex.encode()
    body = _multipart_body(
        boundary,
        file_name="image.jpg",
        file_bytes=image_bytes,
        mime="image/jpeg",
        extra_fields=extra_fields,
    )
    request = urllib.request.Request(
        full_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _http_post_json(url: str, payload: dict[str, Any]) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _call_roboflow_model(
    image_bytes: bytes,
    settings: Settings,
    crop: CropConfig,
) -> tuple[dict[str, Any], int, int]:
    image_bytes, img_w, img_h = _prepare_image_bytes(image_bytes)
    model_id = crop.resolved_model_id()
    url = f"{settings.roboflow_api_url.rstrip('/')}/{model_id}"

    api_conf_pct = max(1, min(40, settings.roboflow_api_confidence_pct))
    overlap_pct = (
        int(round(settings.roboflow_iou_threshold * 100))
        if settings.roboflow_iou_threshold <= 1
        else int(settings.roboflow_iou_threshold)
    )

    for attempt in range(2):
        try:
            payload = _http_post_multipart(
                url,
                image_bytes=image_bytes,
                api_key=settings.roboflow_api_key or "",
                query={"confidence": api_conf_pct, "overlap": overlap_pct},
            )
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Roboflow API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == 0:
                image_bytes, img_w, img_h = _prepare_image_bytes(image_bytes, max_edge=960)
                continue
            raise RuntimeError(f"Roboflow API connection failed: {exc}") from exc

    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise RuntimeError("Unexpected Roboflow response format.")

    img_meta = raw.get("image") or {}
    resp_w = int(img_meta.get("width") or img_w)
    resp_h = int(img_meta.get("height") or img_h)
    return raw, resp_w, resp_h


def _call_roboflow_workflow(
    image_bytes: bytes,
    settings: Settings,
    crop: CropConfig,
) -> tuple[dict[str, Any], int, int]:
    image_bytes, img_w, img_h = _prepare_image_bytes(image_bytes)
    workflow_id = resolve_workflow_id(crop)
    workspace = (crop.workflow_workspace or settings.roboflow_workspace_name).strip("/")
    api_key = settings.roboflow_api_key or ""

    b64 = base64.b64encode(image_bytes).decode("ascii")
    parameters: dict[str, str] = {}
    if crop.workflow_classes:
        parameters["classes"] = crop.workflow_classes

    json_payload = {
        "api_key": api_key,
        "inputs": {"image": {"type": "base64", "value": b64}},
    }
    if parameters:
        json_payload["parameters"] = parameters

    endpoints = [
        f"{settings.roboflow_api_url.rstrip('/')}/{workspace}/workflows/{workflow_id}",
        f"https://detect.roboflow.com/{workspace}/workflows/{workflow_id}",
    ]

    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            payload = _http_post_json(endpoint, json_payload)
            raw = json.loads(payload)
            if isinstance(raw, dict):
                return raw, img_w, img_h
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(
                f"Roboflow workflow error {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
            )
        except (urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(str(last_error))
    raise RuntimeError("Roboflow workflow request failed.")


def _extract_prediction_list(node: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if "class" in node and "confidence" in node:
            out.append(node)
        for value in node.values():
            _extract_prediction_list(value, out)
    elif isinstance(node, list):
        for item in node:
            _extract_prediction_list(item, out)


def _parse_predictions(raw: dict[str, Any]) -> list[dict[str, Any]]:
    preds = raw.get("predictions")
    if isinstance(preds, list):
        return [p for p in preds if isinstance(p, dict)]
    if isinstance(preds, dict):
        nested = preds.get("predictions")
        if isinstance(nested, list):
            return [p for p in nested if isinstance(p, dict)]

    extracted: list[dict[str, Any]] = []
    _extract_prediction_list(raw, extracted)
    if extracted:
        return extracted
    return []


def _has_bbox(pred: dict[str, Any]) -> bool:
    return all(k in pred for k in ("x", "y", "width", "height"))


def _boxes_from_predictions(
    preds: list[dict[str, Any]],
    *,
    img_w: int,
    img_h: int,
    classify_only: bool,
) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    bbox_preds = [p for p in preds if _has_bbox(p)]
    source = bbox_preds if bbox_preds else (preds[:1] if classify_only and preds else preds)

    for p in source:
        if _has_bbox(p):
            boxes.append(
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
            )
        elif classify_only and img_w > 0 and img_h > 0:
            boxes.append(
                DetectionBox(
                    class_name=str(p.get("class") or "unknown"),
                    class_id=str(p.get("class_id")) if p.get("class_id") is not None else None,
                    confidence=round(float(p.get("confidence", 0)) * 100.0, 1),
                    x=img_w / 2.0,
                    y=img_h / 2.0,
                    width=float(img_w) * 0.92,
                    height=float(img_h) * 0.92,
                    color=_color_for_class(str(p.get("class") or "unknown")),
                )
            )
    return boxes


def _primary_class(preds: list[dict[str, Any]]) -> tuple[str, float]:
    if not preds:
        return "unknown", 0.0
    best = max(preds, key=lambda p: float(p.get("confidence", 0)))
    return str(best.get("class", "unknown")), float(best.get("confidence", 0)) * 100.0


def _kb_for_crop(crop: CropConfig, settings: Settings) -> KnowledgeBase:
    path = Path(crop.resolved_classes_path(settings.project_root))
    return KnowledgeBase(path)


def _model_label(crop: CropConfig, settings: Settings) -> str:
    if crop.inference_kind == "workflow":
        return f"workflow:{resolve_workflow_id(crop)}"
    return crop.resolved_model_id()


def _class_matches_crop(class_name: str, crop: CropConfig) -> bool:
    if not crop.validation_keywords:
        return True
    norm = _normalize_label(class_name)
    return any(kw in norm for kw in crop.validation_keywords)


def _assess_crop_match(preds: list[dict[str, Any]], crop: CropConfig) -> tuple[bool, str | None, float]:
    """Return (matches_crop, top_alien_class, top_confidence_pct)."""
    if not preds:
        return True, None, 0.0
    if any(_class_matches_crop(str(p.get("class", "")), crop) for p in preds):
        return True, None, 0.0
    best = max(preds, key=lambda p: float(p.get("confidence", 0)))
    conf = float(best.get("confidence", 0)) * 100.0
    best_class = str(best.get("class", "unknown"))
    if conf < 22.0:
        return True, best_class, conf
    return False, best_class, conf


def validate_crop_leaf(
    image_bytes: bytes,
    *,
    crop_id: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Quick Roboflow check: does this leaf look like the selected crop's model expects?"""
    settings = settings or get_settings()
    if not settings.roboflow_api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set in the environment.")

    crop = get_crop(crop_id)
    image_bytes, _, _ = _prepare_image_bytes(image_bytes, max_edge=640)

    if crop.inference_kind == "workflow":
        raw, _, _ = _call_roboflow_workflow(image_bytes, settings, crop)
    else:
        raw, _, _ = _call_roboflow_model(image_bytes, settings, crop)

    preds = _parse_predictions(raw)
    if not preds:
        return {
            "crop_id": crop.id,
            "status": "no_detection",
            "crop_match": False,
            "top_class": None,
            "top_confidence_pct": 0.0,
            "message": f"No leaf regions detected. Move closer with one {crop.display_name.lower()} leaf in frame.",
        }

    matches, alien_class, alien_conf = _assess_crop_match(preds, crop)
    best = max(preds, key=lambda p: float(p.get("confidence", 0)))
    top_class = str(best.get("class", "unknown"))
    top_conf = round(float(best.get("confidence", 0)) * 100.0, 1)

    if matches and top_conf >= 18:
        return {
            "crop_id": crop.id,
            "status": "match",
            "crop_match": True,
            "top_class": top_class,
            "top_confidence_pct": top_conf,
            "message": f"{crop.display_name} leaf looks good — hold steady.",
        }
    if matches:
        return {
            "crop_id": crop.id,
            "status": "uncertain",
            "crop_match": True,
            "top_class": top_class,
            "top_confidence_pct": top_conf,
            "message": "Leaf detected — move closer or improve lighting.",
        }

    return {
        "crop_id": crop.id,
        "status": "mismatch",
        "crop_match": False,
        "top_class": alien_class or top_class,
        "top_confidence_pct": round(alien_conf or top_conf, 1),
        "message": (
            f"This does not look like a {crop.display_name.lower()} leaf for the selected crop. "
            f"Switch crop tab or retake the photo."
        ),
    }


def _wrong_crop_response(
    crop: CropConfig,
    kb: KnowledgeBase,
    *,
    alien_class: str | None,
    img_w: int,
    img_h: int,
    model_label: str,
    version: str,
) -> DetectResponse:
    unk = kb.get("unknown")
    alien = alien_class or "another crop"
    return DetectResponse(
        result=DetectionResult(
            diseaseName=f"Not a {crop.display_name.lower()} leaf",
            diseaseNameRw=f"Si ikibabi cy'{crop.display_name.lower()}",
            confidence=0.0,
            type="unknown",
            explanation=(
                f"The image looks like a different plant than {crop.display_name} "
                f"(model saw “{alien}”). Select the correct crop tab or upload a matching leaf."
            ),
            explanationRw=(
                f"Ifoto isa n'igihingwa bitandukanye na {crop.display_name} "
                f"(moderi yabonye “{alien}”). Hitamo igihingwa cyo hepfo cyangwa shyiraho ikibabi gikwiye."
            ),
            treatment=unk.treatment,
            treatmentRw=unk.treatmentRw,
            prevention=unk.prevention,
            preventionRw=unk.preventionRw,
            care=unk.care,
            careRw=unk.careRw,
        ),
        model_version=version,
        request_id=str(uuid.uuid4()),
        inference_mode="roboflow",
        crop_id=crop.id,
        top_class_id="wrong_crop",
        rejection_reason="wrong_crop",
        alternatives=[],
        top_confidence_pct=0.0,
        detections=[],
        image_width=img_w,
        image_height=img_h,
        roboflow_model_id=model_label,
    )


def run_roboflow_detect(
    image_bytes: bytes,
    *,
    crop_id: str | None = None,
    settings: Settings | None = None,
    kb: KnowledgeBase | None = None,
) -> DetectResponse:
    settings = settings or get_settings()
    if not settings.roboflow_api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set in the environment.")

    crop = get_crop(crop_id)
    kb = kb or _kb_for_crop(crop, settings)

    if crop.inference_kind == "workflow":
        raw, img_w, img_h = _call_roboflow_workflow(image_bytes, settings, crop)
    else:
        raw, img_w, img_h = _call_roboflow_model(image_bytes, settings, crop)

    preds = _parse_predictions(raw)
    classify_only = crop.inference_kind == "classify"
    boxes = _boxes_from_predictions(preds, img_w=img_w, img_h=img_h, classify_only=classify_only)
    model_label = _model_label(crop, settings)
    version = crop.model_version_label or settings.model_version

    crop_ok, alien_class, _ = _assess_crop_match(preds, crop)
    if preds and not crop_ok:
        return _wrong_crop_response(
            crop,
            kb,
            alien_class=alien_class,
            img_w=img_w,
            img_h=img_h,
            model_label=model_label,
            version=version,
        )

    if not preds:
        unk = kb.get("unknown")
        return DetectResponse(
            result=DetectionResult(
                diseaseName="No disease regions detected",
                diseaseNameRw="Nta bice by'indwara byabonetse",
                confidence=0.0,
                type="unknown",
                explanation=(
                    f"No {crop.display_name.lower()} disease signs were found above the confidence threshold. "
                    "Try a closer photo with visible symptoms."
                ),
                explanationRw=(
                    f"Nta bimenyetso by'indwara bya {crop.display_name.lower()} byabonetse. "
                    "Fata ifoto yegereye igaragaza ibimenyetso."
                ),
                treatment=unk.treatment,
                treatmentRw=unk.treatmentRw,
                prevention=unk.prevention,
                preventionRw=unk.preventionRw,
                care=unk.care,
                careRw=unk.careRw,
            ),
            model_version=version,
            request_id=str(uuid.uuid4()),
            inference_mode="roboflow",
            crop_id=crop.id,
            top_class_id="unknown",
            rejection_reason="low_confidence",
            alternatives=[],
            top_confidence_pct=0.0,
            detections=boxes,
            image_width=img_w,
            image_height=img_h,
            roboflow_model_id=model_label,
        )

    primary_name, primary_conf = _primary_class(preds)
    kb_id = _map_class_to_kb(primary_name, kb, crop)
    primary_norm = _normalize_label(primary_name)
    result_type: str = "disease"
    if "healthy" in primary_norm and primary_conf >= 50:
        result_type = "healthy"

    alternatives: list[ClassAlternative] = []
    seen: set[str] = set()
    for p in sorted(preds, key=lambda x: float(x.get("confidence", 0)), reverse=True):
        cname = str(p.get("class", ""))
        if cname in seen:
            continue
        seen.add(cname)
        cid = _map_class_to_kb(cname, kb, crop) or cname
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
        if result_type == "healthy":
            result.type = "healthy"  # type: ignore[misc]
        elif result_type == "disease":
            result.type = "disease"  # type: ignore[misc]
    else:
        unk = kb.get("unknown")
        result = DetectionResult(
            diseaseName=primary_name,
            diseaseNameRw=primary_name,
            confidence=round(primary_conf, 1),
            type=result_type,  # type: ignore[arg-type]
            explanation=(
                f"Detected {primary_name} on {len(preds)} region(s) for {crop.display_name}. "
                "Confirm with a local agronomist."
            ),
            explanationRw=(
                f"Byabonetse {primary_name} mu bice {len(preds)} kuri {crop.display_name}. "
                "Emeza n'umuhinzi w'impuguke."
            ),
            treatment=unk.treatment,
            treatmentRw=unk.treatmentRw,
            prevention=unk.prevention,
            preventionRw=unk.preventionRw,
            care=unk.care,
            careRw=unk.careRw,
        )
        kb_id = primary_name

    return DetectResponse(
        result=result,
        model_version=version,
        request_id=str(uuid.uuid4()),
        inference_mode="roboflow",
        crop_id=crop.id,
        top_class_id=kb_id,
        rejection_reason=None,
        alternatives=alternatives[:12],
        top_confidence_pct=round(primary_conf, 1),
        detections=boxes,
        image_width=img_w,
        image_height=img_h,
        roboflow_model_id=model_label,
    )
