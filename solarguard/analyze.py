"""FR4–FR5 — hand-crafted features + hand-written rules. One entry point: analyze(img)."""
import cv2
import numpy as np

from .dirt import dirt_mask, preprocess
from .panel import panel_mask

LABELS = ["Temiz", "Tozlu", "Kuş pisliği", "Kar", "Anomali"]
KAGGLE_TO_LABEL = {
    "Clean": "Temiz", "Dusty": "Tozlu", "Bird-drop": "Kuş pisliği", "Snow-Covered": "Kar",
    "Electrical-damage": "Anomali", "Physical-Damage": "Anomali",
}

# Thresholds picked by reading per-class feature percentiles on the dev split
# (scripts/eval_dataset.py --stats). Not learned by any algorithm.
RULES = {
    "snow": 0.10,       # white-area fraction of the whole image
    "glcm_hi": 40,      # GLCM contrast; cracks and burn marks make it high
    "edges": 0.34,      # Canny edge density on the panel (cracked glass)
    "rust": 0.03,       # orange/brown fraction of the panel (burn marks, rust)
    "blobs": 9,         # compact white blobs on the panel (droppings)
    "blob_sat": 45,     # ...on a panel that is not grey all over
    "sat": 55,          # mean panel saturation; dusty panels look grey
    "glcm_lo": 10,      # dust film smooths texture -> low GLCM contrast
    "glcm_lo_sat": 80,  # ...and greys the panel; texture alone is too light-sensitive
    "dirt": 0.25,       # dirt ratio of the panel
}
DIRTY = {"Tozlu", "Kuş pisliği", "Kar"}   # labels that mean "needs cleaning"


def glcm_contrast(gray, levels=32):
    """GLCM contrast for horizontal neighbours, in plain NumPy (no ML library)."""
    q = (gray.astype(np.uint16) * levels // 256)
    a, b = q[:, :-1].ravel(), q[:, 1:].ravel()
    p = np.bincount(a * levels + b, minlength=levels * levels).astype(np.float64)
    p /= max(p.sum(), 1)
    i, j = np.divmod(np.arange(levels * levels), levels)
    return float((p * (i - j) ** 2).sum())


def white_blobs(hsv, panel):
    white = ((hsv[..., 2] > 190) & (hsv[..., 1] < 60) & (panel > 0)).astype(np.uint8)
    # Opening strips the thin white grid lines so droppings touching them stay separate blobs.
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white)
    area = panel.size
    count = 0
    for x, y, w, h, a in stats[1:]:
        if 0.0003 * area <= a <= 0.02 * area and max(w, h) <= 3 * min(w, h) and a >= 0.4 * w * h:
            count += 1
    return count


def features(img, panel=None):
    """panel: a fixed ROI (Live Mode: a fixed camera is calibrated once on the clean panel)."""
    if panel is None:
        panel = panel_mask(img)
    if np.count_nonzero(panel) < 0.10 * panel.size:   # fallback: no panel found -> whole image
        panel = np.full(panel.shape, 255, np.uint8)
    clean, hsv = preprocess(img)
    dirt, glare, ratio = dirt_mask(hsv, panel)
    inside = panel > 0
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    white = ((val > 200) & (sat < 40)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    rust = (hue >= 5) & (hue <= 25) & (sat > 90) & (val > 60) & inside
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200) > 0
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Contrast stretch (2–98 percentile) so GLCM measures texture, not how bright the sun is.
    lo, hi = np.percentile(small, (2, 98))
    small = np.clip((small - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    f = {
        "coverage": float(inside.mean()),
        "dirt": float(ratio),
        "snow": float(white.mean()),
        "rust": float(rust.sum() / inside.sum()),
        "edges": float(edges[inside].mean()),
        "blobs": white_blobs(hsv, panel),
        "sat": float(sat[inside].mean()),
        "val": float(val[inside].mean()),
        "glcm": glcm_contrast(small),
    }
    return f, {"panel": panel, "dirt": dirt, "glare": glare, "pre": clean}


def classify(f, r=RULES):
    """Hand-written decision rules. Returns (label, human-readable reason)."""
    if f["snow"] > r["snow"]:
        return "Kar", f"white area {f['snow']:.0%} > {r['snow']:.0%}"
    if f["glcm"] > r["glcm_hi"]:
        return "Anomali", f"high texture contrast (GLCM {f['glcm']:.0f})"
    if f["edges"] > r["edges"]:
        return "Anomali", f"edge density {f['edges']:.0%} (cracks?)"
    if f["rust"] > r["rust"]:
        return "Anomali", f"brown/burn colour {f['rust']:.1%} of panel"
    if f["blobs"] >= r["blobs"] and f["sat"] >= r["blob_sat"]:
        return "Kuş pisliği", f"{f['blobs']} compact white spots"
    if f["sat"] < r["sat"]:
        return "Tozlu", f"grey panel (mean saturation {f['sat']:.0f})"
    if f["glcm"] < r["glcm_lo"] and f["sat"] < r["glcm_lo_sat"]:
        return "Tozlu", f"smoothed texture (GLCM {f['glcm']:.1f})"
    if f["dirt"] > r["dirt"]:
        return "Tozlu", f"dirt {f['dirt']:.0%} > {r['dirt']:.0%}"
    return "Temiz", f"dirt {f['dirt']:.0%}, saturation {f['sat']:.0f}"


def analyze(img, rules=RULES, panel=None):
    f, masks = features(img, panel)
    label, reason = classify(f, rules)
    return {"label": label, "reason": reason, "features": f}, masks


def overlay(img, masks):
    """Visual for UI/report: dirt in red, glare in yellow, panel outline in green."""
    out = img.copy()
    out[masks["dirt"] > 0] = (0, 0, 255)
    out[masks["glare"] > 0] = (0, 255, 255)
    out = cv2.addWeighted(img, 0.55, out, 0.45, 0)
    contours, _ = cv2.findContours(masks["panel"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out
