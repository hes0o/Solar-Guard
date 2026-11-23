"""FR1 — panel localization (classical image processing only).

Dataset images are mostly angled close-ups of whole arrays with sky, roofs, trees
or snow around them, so instead of looking for a single quadrilateral we find
panel pixels by colour (blue or dark cells) + texture (cell grid lines), keep the
large regions, and fill each region's convex hull so dirt/snow/droppings lying
*on* the panel stay inside the ROI.
"""
import cv2
import numpy as np

MAX_SIDE = 640


def load(path, max_side=MAX_SIDE):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def panel_mask(img, min_area=0.02):
    """Return a uint8 mask (255 = panel) for a BGR image."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # Colour: blue cells (polycrystalline) or dark cells (monocrystalline).
    blue = (hue >= 95) & (hue <= 130) & (sat >= 50) & (val >= 40)
    # Dusty mono panels turn grey, so "dark" also admits low-saturation mid-greys.
    dark = ((val < 70) & (sat < 120)) | ((val < 140) & (sat < 60))

    # Texture: cell grid lines give local contrast; clear sky does not.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = (15, 15)
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, k) - cv2.blur(gray, k) ** 2, 0))
    textured = std > 8

    cand = ((blue | dark) & textured).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    mask = np.zeros((h, w), np.uint8)
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= min_area * h * w:
            cv2.fillPoly(mask, [cv2.convexHull(c)], 255)
    return mask


def coverage(mask):
    return float(np.count_nonzero(mask)) / mask.size
