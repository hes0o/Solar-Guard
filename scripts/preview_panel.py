"""Visual check for FR1: random samples per class with the panel mask outlined.

usage: python scripts/preview_panel.py [n_per_class] [seed]  -> out/panel_preview.jpg
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarguard.panel import coverage, load, panel_mask  # noqa: E402

ROOT = Path("data/raw/Faulty_solar_panel")
EXTS = {".jpg", ".jpeg", ".png"}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)

rows = []
for cls in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    files = [f for f in cls.rglob("*") if f.suffix.lower() in EXTS]
    tiles = []
    for f in random.sample(files, n):
        img = load(f)
        mask = panel_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        tile = cv2.resize(img, (240, 180))
        cv2.putText(tile, f"{cls.name[:10]} {coverage(mask):.0%}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        tiles.append(tile)
    rows.append(np.hstack(tiles))

Path("out").mkdir(exist_ok=True)
cv2.imwrite("out/panel_preview.jpg", np.vstack(rows))
print("wrote out/panel_preview.jpg")
