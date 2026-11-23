import numpy as np

from solarguard.panel import coverage, panel_mask


def synthetic_panel():
    """Blue gridded panel with a white 'dropping' on it, on a flat beige background."""
    img = np.full((300, 400, 3), (170, 200, 220), np.uint8)  # beige (BGR)
    img[60:240, 80:320] = (160, 70, 20)                        # blue cells
    img[60:240:20, 80:320] = 255                               # grid rows
    img[60:240, 80:320:20] = 255                               # grid cols
    img[130:160, 180:220] = 245                                # dirt blob on panel
    return img


def test_finds_panel_and_keeps_dirt_inside():
    mask = panel_mask(synthetic_panel())
    panel = mask[60:240, 80:320]
    assert np.count_nonzero(panel) / panel.size > 0.9
    assert mask[130:160, 180:220].all()          # dirt stays inside the ROI
    outside = mask.copy()
    outside[50:250, 70:330] = 0                   # small margin for morphology
    assert np.count_nonzero(outside) / outside.size < 0.02


def test_empty_scene_has_no_panel():
    assert coverage(panel_mask(np.full((200, 200, 3), 200, np.uint8))) == 0
