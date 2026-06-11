"""Reject obvious non-leaf uploads before the classifier runs.



The crop model only knows trained disease classes. On photos of people, cities, or

documents it still picks a disease with very high softmax scores. Rules here score

how leaf-like an upload is using color, layout, and scene cues.



Analysis uses a center crop so black/white borders (common on web images) do not dominate.

"""



from __future__ import annotations



import numpy as np

from PIL import Image





def _center_crop_pixels(px: np.ndarray, inset: float = 0.14) -> np.ndarray:

    h, w = px.shape[:2]

    dx = int(w * inset)

    dy = int(h * inset)

    if w <= 2 * dx or h <= 2 * dy:

        return px

    return px[dy : h - dy, dx : w - dx]





def _inner_crop_pixels(px: np.ndarray, inset: float = 0.28) -> np.ndarray:

    h, w = px.shape[:2]

    dx = int(w * inset)

    dy = int(h * inset)

    if w <= 2 * dx or h <= 2 * dy:

        return px

    return px[dy : h - dy, dx : w - dx]





def _skin_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray, lum: np.ndarray) -> np.ndarray:

    """Human skin — stricter than before so sand/soil/wood are not matched."""

    return (

        (r > 105)

        & (g > 55)

        & (b > 35)

        & (r > g + 22)

        & (r > b + 35)

        & (np.abs(r - g) > 22)

        & (g > b)

        & (lum >= 85)

        & (lum <= 210)

    )





def _leaf_color_masks(px: np.ndarray) -> dict[str, np.ndarray]:

    r = px[..., 0]

    g = px[..., 1]

    b = px[..., 2]

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    chroma = np.max(px, axis=-1) - np.min(px, axis=-1)

    skin = _skin_mask(r, g, b, lum)
    sunset_sky = (r > 155) & (g > 75) & (b < 105) & (r > g + 25)

    green = (g > r + 8) & (g > b + 8) & (g >= 45) & (lum >= 40) & (lum <= 220)

    chlorotic = (g >= 50) & (r >= 60) & (lum >= 45) & (lum <= 238) & (chroma >= 14)

    yellow_leaf = (r >= 90) & (g >= 75) & (b <= 155) & (lum >= 50) & (lum <= 238) & (chroma >= 16)

    warm_leaf = (

        (r >= g - 8)

        & (r >= 35)

        & (lum >= 38)

        & (lum <= 215)

        & (chroma >= 16)

        & (chroma <= 95)

        & ~((b > r + 28) & (b > g + 20))

    )

    leaf = (green | chlorotic | yellow_leaf | warm_leaf) & ~skin & ~sunset_sky



    sky = (b > r + 12) & (b > g + 4) & (lum >= 95) & (lum <= 245)

    neutral_scene = (chroma < 22) & (lum >= 55) & (lum <= 210)



    return {

        "leaf": leaf,

        "green": green,

        "chlorotic": chlorotic,

        "yellow_leaf": yellow_leaf,

        "skin": skin,

        "sky": sky,

        "neutral_scene": neutral_scene,

        "lum": lum,

        "chroma": chroma,

    }





def _largest_blob_fraction(mask: np.ndarray) -> float:

    h, w = mask.shape

    if not mask.any():

        return 0.0



    visited = np.zeros_like(mask, dtype=bool)

    best = 0

    ys, xs = np.where(mask)

    for y0, x0 in zip(ys, xs):

        if visited[y0, x0]:

            continue

        stack = [(int(y0), int(x0))]

        visited[y0, x0] = True

        size = 0

        while stack:

            y, x = stack.pop()

            size += 1

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):

                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:

                    visited[ny, nx] = True

                    stack.append((ny, nx))

        if size > best:

            best = size

    return float(best) / float(mask.size)





def _leaf_color_metrics(px: np.ndarray) -> dict[str, float]:

    masks = _leaf_color_masks(px)

    lum = masks["lum"]

    chroma = masks["chroma"]

    r, g, b = px[..., 0], px[..., 1], px[..., 2]



    dark_frac = float((lum < 45).mean())

    bright_frac = float((lum > 215).mean())

    foliage_frac = float(((lum >= 45) & (lum <= 235)).mean())

    mid_frac = float(((lum >= 60) & (lum <= 230)).mean())

    mean_chroma = float(chroma.mean())



    leaf_frac = float(masks["leaf"].mean())

    green_frac = float(masks["green"].mean())

    chlorotic_frac = float(masks["chlorotic"].mean())

    yellow_frac = float(masks["yellow_leaf"].mean())

    skin_frac = float(masks["skin"].mean())

    sky_frac = float(masks["sky"].mean())

    neutral_frac = float(masks["neutral_scene"].mean())

    leaf_blob_frac = _largest_blob_fraction(masks["leaf"])



    organic = (chroma >= 18) & (lum >= 35) & (lum <= 235)

    organic = organic & ~((b > r + 28) & (b > g + 20))

    organic_frac = float(organic.mean())



    warm = (r >= g - 5) & (r >= 40) & (lum >= 40) & (lum <= 220) & (chroma >= 15)

    warm_frac = float(warm.mean())



    ui_blue = (b > r + 25) & (b > g + 15) & (b > 90)

    ui_blue_frac = float(ui_blue.mean())



    return {

        "dark_frac": dark_frac,

        "bright_frac": bright_frac,

        "foliage_frac": foliage_frac,

        "mid_frac": mid_frac,

        "mean_chroma": mean_chroma,

        "leaf_frac": leaf_frac,

        "green_frac": green_frac,

        "chlorotic_frac": chlorotic_frac,

        "yellow_frac": yellow_frac,

        "skin_frac": skin_frac,

        "sky_frac": sky_frac,

        "neutral_frac": neutral_frac,

        "leaf_blob_frac": leaf_blob_frac,

        "organic_frac": organic_frac,

        "warm_frac": warm_frac,

        "ui_blue_frac": ui_blue_frac,

    }





def _center_vs_outer_leaf_ratio(px: np.ndarray) -> float:

    h, w = px.shape[:2]

    cy0, cy1 = int(h * 0.28), int(h * 0.72)

    cx0, cx1 = int(w * 0.28), int(w * 0.72)

    inner = px[cy0:cy1, cx0:cx1]

    masks = _leaf_color_masks(px)

    inner_leaf = float(_leaf_color_masks(inner)["leaf"].mean())

    outer_mask = np.ones((h, w), dtype=bool)

    outer_mask[cy0:cy1, cx0:cx1] = False

    outer_leaf = float(masks["leaf"][outer_mask].mean()) if outer_mask.any() else 0.0

    if outer_leaf < 0.02:

        return inner_leaf / 0.02

    return inner_leaf / outer_leaf





def is_obvious_scene_photo(image: Image.Image) -> bool:
    """
    Reject panoramas, posters, meeting rooms, and cityscapes on the full frame
    before auto-cropping can isolate a misleading green region.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    aspect = w / max(h, 1)

    small = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    px = np.asarray(small, dtype=np.float32)
    m = _leaf_color_metrics(px)

    top = px[: max(1, px.shape[0] // 3), :]
    top_m = _leaf_color_metrics(top)
    tr, tg, tb = top[..., 0], top[..., 1], top[..., 2]
    tlum = 0.2126 * tr + 0.7152 * tg + 0.0722 * tb
    sunset = (tr > 170) & (tg > 90) & (tb < 130) & (tlum >= 110)
    sunset_frac = float(sunset.mean())

    if top_m["sky_frac"] >= 0.10 and m["green_frac"] >= 0.18 and m["leaf_blob_frac"] < 0.14:
        return True
    if sunset_frac >= 0.12 and m["green_frac"] >= 0.18 and m["leaf_blob_frac"] < 0.14:
        return True
    if m["skin_frac"] >= 0.08 and m["leaf_blob_frac"] < 0.08:
        return True

    # Portrait poster / cityscape (sky band on top, not a leaf photo)
    if aspect < 0.85 and sunset_frac >= 0.30:
        return True
    if aspect < 0.85 and top_m["sky_frac"] >= 0.18:
        return True

    # Tall panorama: widespread green but no dominant isolated leaf blob
    if aspect < 0.82 and m["green_frac"] >= 0.20 and m["leaf_blob_frac"] < 0.12:
        return True

    return False


def leaf_plausibility_score(image: Image.Image) -> float:

    """Return 0–1 score for how likely the image is a close-up crop leaf photo."""

    rgb = image.convert("RGB")

    small = rgb.resize((224, 224), Image.Resampling.BILINEAR)

    px = np.asarray(small, dtype=np.float32)

    center = _center_crop_pixels(px)

    inner = _inner_crop_pixels(px)



    m = _leaf_color_metrics(center)

    inner_m = _leaf_color_metrics(inner)

    focus_ratio = _center_vs_outer_leaf_ratio(px)



    # Fast accept: dominant leaf blob in center (soil/black/white backgrounds OK)

    if inner_m["leaf_frac"] >= 0.16 and inner_m["leaf_blob_frac"] >= 0.05:

        return 0.85

    if inner_m["leaf_frac"] >= 0.10 and inner_m["leaf_blob_frac"] >= 0.08:

        return 0.72



    score = 0.0

    score += min(0.34, inner_m["leaf_frac"] * 1.15)

    score += min(0.22, inner_m["leaf_blob_frac"] * 1.8)

    score += min(0.14, max(0.0, (focus_ratio - 0.9) * 0.08))



    if inner_m["green_frac"] >= 0.10:

        score += 0.10

    if inner_m["chlorotic_frac"] >= 0.12 or inner_m["yellow_frac"] >= 0.10:

        score += 0.10

    if inner_m["leaf_frac"] >= 0.22 and inner_m["leaf_blob_frac"] >= 0.10:

        score += 0.12



    # Scene penalties — only when leaf signal is weak in the center

    if inner_m["leaf_frac"] < 0.12:

        if m["skin_frac"] >= 0.07:

            score -= 0.42

        if m["sky_frac"] >= 0.10:

            score -= 0.30

        if m["neutral_frac"] >= 0.34 and inner_m["leaf_blob_frac"] < 0.08:

            score -= 0.22

        if inner_m["green_frac"] >= 0.08 and inner_m["leaf_blob_frac"] < 0.06 and focus_ratio < 1.15:

            score -= 0.28



    if m["ui_blue_frac"] > 0.16 and inner_m["leaf_frac"] < 0.10:

        score -= 0.35

    if m["dark_frac"] > 0.12 and m["bright_frac"] > 0.03 and m["mean_chroma"] < 35:

        score -= 0.40

    if m["mean_chroma"] < 14:

        score -= 0.50



    return float(np.clip(score, 0.0, 1.0))





def looks_like_crop_leaf_photo(image: Image.Image, *, min_score: float = 0.42) -> bool:

    """

    Return True only if the image plausibly contains a close-up crop leaf.



    Designed to reject: people, cityscapes, terminals, documents, meeting rooms.

    Allows: leaves on soil, sand, black studio backgrounds, diseased/yellow foliage.

    """

    rgb = image.convert("RGB")

    small = rgb.resize((224, 224), Image.Resampling.BILINEAR)

    px = np.asarray(small, dtype=np.float32)

    center = _center_crop_pixels(px)

    m = _leaf_color_metrics(center)

    inner = _inner_crop_pixels(px)

    inner_m = _leaf_color_metrics(inner)



    # Strong leaf in center — allow immediately (training + field photos)

    if inner_m["leaf_frac"] >= 0.14 and inner_m["leaf_blob_frac"] >= 0.05:

        return True



    if m["skin_frac"] >= 0.09 and inner_m["leaf_blob_frac"] < 0.08:

        return False

    if m["sky_frac"] >= 0.12 and inner_m["leaf_blob_frac"] < 0.08:

        return False

    if m["dark_frac"] > 0.12 and m["bright_frac"] > 0.03 and m["mean_chroma"] < 35:

        return False

    if m["bright_frac"] > 0.48 and m["mid_frac"] < 0.20:

        return False

    if m["mean_chroma"] < 14:

        return False

    if m["ui_blue_frac"] > 0.16 and inner_m["leaf_frac"] < 0.10:

        return False



    focus_ratio = _center_vs_outer_leaf_ratio(px)

    if (

        inner_m["green_frac"] >= 0.08

        and inner_m["leaf_blob_frac"] < 0.06

        and focus_ratio < 1.15

        and inner_m["leaf_frac"] < 0.12

    ):

        return False



    return leaf_plausibility_score(image) >= min_score


