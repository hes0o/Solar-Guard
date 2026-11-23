"""FR2–FR3 — preprocessing and dirt segmentation (classical only).

Clean cells are dark or saturated blue; dust, droppings and snow are bright and
desaturated. So each pixel gets a "dirt index" D = V * (1 - S), thresholded
inside the panel ROI. Thin white grid lines/busbars score high too, so a
morphological opening removes anything thinner than the kernel.
"""
import cv2
import numpy as np

DIRT_THRESHOLD = 100   # min D for a pixel to count as dirt (0–255)
GLARE_V = 250          # specular highlights: excluded from the dirt ratio


def preprocess(img):
    """Denoise + CLAHE on lightness. Returns (BGR image, HSV image)."""
    img = cv2.bilateralFilter(img, 5, 40, 40)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def dirt_index(hsv):
    s = hsv[..., 1].astype(np.float32) / 255
    return (hsv[..., 2].astype(np.float32) * (1 - s)).astype(np.uint8)


def drop_elongated(mask, max_aspect=4.0, max_width=25):
    """Remove long thin components: aluminium frames and gaps between modules, not dirt."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        short, long_ = min(w, h), max(w, h)
        fill = cv2.contourArea(c) / max(w * h, 1)
        if short <= max_width and long_ >= max_aspect * max(short, 1) and fill > 0.4:
            cv2.drawContours(mask, [c], -1, 0, -1)
    return mask


def dirt_mask(hsv, panel, threshold=DIRT_THRESHOLD):
    """Return (dirt mask, glare mask, dirt ratio of the usable panel area)."""
    d = dirt_index(hsv)
    # Otsu adapts to each image, but never below the fixed floor (clean panels would split anyway).
    roi = d[panel > 0]
    t = max(threshold, cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]) if roi.size else threshold
    mask = ((d > t) & (panel > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = drop_elongated(mask)
    glare = ((hsv[..., 2] >= GLARE_V) & (hsv[..., 1] < 25) & (panel > 0)).astype(np.uint8) * 255
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask[glare > 0] = 0
    usable = np.count_nonzero(panel) - np.count_nonzero(glare)
    return mask, glare, (np.count_nonzero(mask) / usable if usable > 0 else 0.0)
