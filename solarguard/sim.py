"""FR9–FR10 — digital twin of one panel: dust build-up, bird droppings, daylight, power.

Dirt is painted onto a real clean Kaggle photo with image-processing operations
(blurred noise texture + alpha blending), so the pipeline has to *see* it.
Power is computed from the twin's true soiling, not from what the pipeline
measured, so the two can be compared.
"""
import math

import cv2
import numpy as np

from .panel import panel_mask

DUST_BGR = np.array([150, 168, 180], np.float32)   # beige-grey dust
P_NOMINAL_W = 400.0


def irradiance(hour):
    """Clear-sky curve, 0 at night, 1 at solar noon (sunrise 06:00, sunset 19:00)."""
    return max(0.0, math.sin(math.pi * (hour - 6) / 13)) if 6 <= hour <= 19 else 0.0


class Twin:
    def __init__(self, pid, base, rng):
        self.pid, self.base, self.rng = pid, base, rng
        self.panel = panel_mask(base)
        h, w = base.shape[:2]
        noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 12)
        self.texture = cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)   # fixed dust pattern per panel
        self.dust_per_day = rng.uniform(0.04, 0.12)                          # each panel soils at its own pace
        self.dust = 0.0
        self.drops = []                                                      # (x, y, rx, ry)

    def step(self, hours, drop=False):
        self.dust = min(1.0, self.dust + self.dust_per_day * hours / 24 * self.rng.uniform(0.5, 1.5))
        if drop or self.rng.random() < 0.05 * hours:                         # ~1 dropping a day
            ys, xs = np.nonzero(self.panel)
            if len(xs):
                i = self.rng.integers(len(xs))
                self.drops.append((int(xs[i]), int(ys[i]), int(self.rng.integers(6, 16)), int(self.rng.integers(5, 12))))

    def clean(self):
        self.dust, self.drops = 0.0, []

    @property
    def soiling(self):
        """True fraction of light blocked, for the power model."""
        return min(1.0, 0.6 * self.dust + 0.01 * len(self.drops))

    def render(self, irr):
        img = self.base.astype(np.float32)
        a = np.clip(self.dust * (0.4 + self.texture), 0, 0.9) * (self.panel > 0)
        img = img * (1 - a[..., None]) + DUST_BGR * a[..., None]
        if self.drops:
            layer = np.zeros(self.panel.shape, np.uint8)
            for x, y, rx, ry in self.drops:
                cv2.ellipse(layer, (x, y), (rx, ry), 0, 0, 360, 255, -1)
            b = cv2.GaussianBlur(layer, (0, 0), 2).astype(np.float32)[..., None] / 255
            img = img * (1 - b) + 240 * b
        img *= 0.35 + 0.65 * irr                                             # daylight
        return np.clip(img, 0, 255).astype(np.uint8)

    def power(self, irr):
        noise = self.rng.normal(0, 0.01)
        return max(0.0, P_NOMINAL_W * irr * (1 - 0.5 * self.soiling) * (1 + noise))
