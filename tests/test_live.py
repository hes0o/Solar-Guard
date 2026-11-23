from datetime import datetime

import numpy as np
import pytest

from solarguard import live as live_mod
from solarguard.live import Live


def make(monkeypatch, labels, mode="manual"):
    """Live with one blank panel, noon, and analyze() replaced by a scripted label sequence."""
    seq = iter(labels)
    monkeypatch.setattr(live_mod, "analyze",
                        lambda img, panel=None: ({"label": next(seq), "reason": "test", "features": {"dirt": 0.5}},
                                                 {"panel": panel, "dirt": panel * 0, "glare": panel * 0}))
    monkeypatch.setattr(live_mod, "overlay", lambda img, masks: img)
    lv = Live([np.zeros((60, 80, 3), np.uint8)], start=datetime(2026, 6, 1, 11, 0))
    lv.settings.update(mode=mode, k=3, minutes_per_tick=10)
    return lv


def tasks(lv):
    return lv.snapshot()["tasks"]


def test_needs_k_consecutive_readings(monkeypatch):
    lv = make(monkeypatch, ["Tozlu", "Tozlu", "Temiz", "Tozlu", "Tozlu", "Tozlu"])
    for _ in range(5):
        lv.tick()
    assert tasks(lv) == []                       # streak was broken by "Temiz"
    lv.tick()
    assert [t["status"] for t in tasks(lv)] == ["suggested"]


def test_manual_flow_approve_start_done_cleans_panel(monkeypatch):
    lv = make(monkeypatch, ["Tozlu"] * 10)
    for _ in range(3):
        lv.tick()
    tid = tasks(lv)[0]["id"]
    lv.twins["P-01"].dust = 0.9
    for action in ("approve", "start", "done"):
        lv.act(tid, action)
    assert tasks(lv)[0]["status"] == "done"
    assert lv.twins["P-01"].dust == 0
    for _ in range(3):                            # cooldown: no new task right away
        lv.tick()
    assert len(tasks(lv)) == 1
    assert lv.snapshot()["unread"]["staff"] == 1  # the approval notified staff


def test_auto_mode_goes_straight_to_staff(monkeypatch):
    lv = make(monkeypatch, ["Anomali"] * 3, mode="auto")
    for _ in range(3):
        lv.tick()
    t = tasks(lv)[0]
    assert (t["kind"], t["status"]) == ("inspection", "new")
    assert lv.snapshot()["unread"] == {"manager": 0, "staff": 1}


def test_state_survives_restart(monkeypatch, tmp_path):
    db = str(tmp_path / "sg.db")
    monkeypatch.setattr(live_mod, "analyze",
                        lambda img, panel=None: ({"label": "Tozlu", "reason": "t", "features": {"dirt": 0.5}},
                                                 {"panel": panel, "dirt": panel * 0, "glare": panel * 0}))
    monkeypatch.setattr(live_mod, "overlay", lambda img, masks: img)
    img = np.zeros((60, 80, 3), np.uint8)
    lv = Live([img], db, start=datetime(2026, 6, 1, 11, 0), base_ids=["Clean/x.jpg"])
    lv.update_settings({"mode": "auto", "k": 2})
    lv.twins["P-01"].dust = 0.4
    lv.tick()
    lv.tick()                                     # second reading -> task created in auto mode
    lv.db.close()

    data = live_mod.saved_state(db)
    assert data["base_ids"] == ["Clean/x.jpg"]
    lv2 = Live([img], db)
    lv2.restore(data)
    assert lv2.t == lv.t and lv2.settings["mode"] == "auto" and lv2.settings["k"] == 2
    assert lv2.twins["P-01"].dust == pytest.approx(lv.twins["P-01"].dust)
    assert lv2.state["P-01"]["streak"] == 2
    assert [t["status"] for t in lv2.snapshot()["tasks"]] == ["new"]
    assert len(lv2.history("P-01")) == 2


def test_bad_settings_change_nothing(monkeypatch):
    lv = make(monkeypatch, [])
    with pytest.raises(ValueError):
        lv.update_settings({"k": 5, "mode": "turbo"})
    assert lv.settings["k"] == 3


def test_invalid_transition_and_settings_rejected(monkeypatch):
    lv = make(monkeypatch, ["Tozlu"] * 3)
    for _ in range(3):
        lv.tick()
    with pytest.raises(ValueError):
        lv.act(tasks(lv)[0]["id"], "done")        # suggested -> done is not allowed
    with pytest.raises(ValueError):
        lv.update_settings({"mode": "turbo"})
    with pytest.raises(ValueError):
        lv.update_settings({"k": 0})
