"""FR9–FR17 — Live Mode: virtual panels, decision engine, tasks, notifications, history.

One Live object owns all state. tick() advances the simulated clock; the web
layer calls it from a background loop and reads state through snapshot().
"""
import sqlite3
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np

from .analyze import DIRTY, analyze, overlay
from .sim import Twin, irradiance

OPEN = ("suggested", "new", "started")
SCHEMA = """
create table if not exists readings(id integer primary key, panel text, t text, irr real, power real,
    dirt real, soiling real, label text);
create table if not exists tasks(id integer primary key, panel text, kind text, status text, reason text,
    created text, updated text);
create table if not exists notifications(id integer primary key, role text, text text, t text, read int default 0);
"""


class Live:
    def __init__(self, bases, db_path=":memory:", seed=0, start=datetime(2026, 6, 1, 6, 0)):
        rng = np.random.default_rng(seed)
        self.twins = {f"P-{i + 1:02d}": Twin(f"P-{i + 1:02d}", b, rng) for i, b in enumerate(bases)}
        self.t = start
        # shortcut: settings live in memory and reset on restart; persist them if that starts to matter.
        self.settings = {"mode": "manual", "minutes_per_tick": 30, "tick_seconds": 3, "k": 3,
                         "cooldown_hours": 24, "price_tl_kwh": 3.0}
        self.state = {pid: {"streak": 0, "kind": None, "cooldown_until": start, "last": None,
                            "frame": None, "overlay": None, "lost_kwh": 0.0, "power": 0.0, "irr": 0.0}
                      for pid in self.twins}
        self.lock = threading.Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---- simulation ---------------------------------------------------------
    def tick(self):
        hours = self.settings["minutes_per_tick"] / 60
        t = self.t + timedelta(hours=hours)
        irr = irradiance(t.hour + t.minute / 60)
        results = {}
        for pid, tw in self.twins.items():          # heavy image work outside the lock
            tw.step(hours)
            frame = tw.render(irr)
            res = masks = None
            if irr >= 0.3:                          # analyse only in good light (no dawn/dusk/night frames)
                res, masks = analyze(frame, panel=tw.panel)
            results[pid] = (frame, res, masks, tw.power(irr), tw.soiling)
        with self.lock:
            self.t = t
            for pid, (frame, res, masks, power, soiling) in results.items():
                st = self.state[pid]
                st["frame"] = jpg(frame)
                st["power"], st["irr"] = power, irr
                st["lost_kwh"] += 0.4 * irr * 0.5 * soiling * hours   # 400 W panel, same loss model as Twin.power
                if res is None:
                    continue
                st["last"], st["overlay"] = res, jpg(overlay(frame, masks))
                self.db.execute("insert into readings(panel,t,irr,power,dirt,soiling,label) values(?,?,?,?,?,?,?)",
                                (pid, iso(t), irr, power, res["features"]["dirt"], soiling, res["label"]))
                self._decide(pid, res)
            self.db.commit()

    # ---- FR11 decision engine ----------------------------------------------
    def _decide(self, pid, res):
        st, label = self.state[pid], res["label"]
        kind = "cleaning" if label in DIRTY else "inspection" if label == "Anomali" else None
        st["streak"] = st["streak"] + 1 if kind and kind == st["kind"] else (1 if kind else 0)
        st["kind"] = kind
        if not kind or st["streak"] < self.settings["k"] or self.t < st["cooldown_until"]:
            return
        if self.db.execute(f"select 1 from tasks where panel=? and status in {OPEN}", (pid,)).fetchone():
            return
        reason = f"{label} for {st['streak']} readings in a row — {res['reason']}"
        if self.settings["mode"] == "auto":
            self._task(pid, kind, "new", reason)
            self._notify("staff", f"{pid}: new {kind} task ({label})")
        else:
            self._task(pid, kind, "suggested", reason)
            self._notify("manager", f"{pid}: {kind} suggested ({label}) — approve or reject")

    def _task(self, pid, kind, status, reason):
        self.db.execute("insert into tasks(panel,kind,status,reason,created,updated) values(?,?,?,?,?,?)",
                        (pid, kind, status, reason, iso(self.t), iso(self.t)))

    def _notify(self, role, text):
        self.db.execute("insert into notifications(role,text,t) values(?,?,?)", (role, text, iso(self.t)))

    # ---- FR13–FR15 task workflow --------------------------------------------
    TRANSITIONS = {"approve": ("suggested", "new"), "reject": ("suggested", "rejected"),
                   "start": ("new", "started"), "done": ("started", "done")}

    def act(self, task_id, action):
        if action not in self.TRANSITIONS:
            raise ValueError("unknown action")
        frm, to = self.TRANSITIONS[action]
        with self.lock:
            row = self.db.execute("select * from tasks where id=?", (task_id,)).fetchone()
            if not row or row["status"] != frm:
                raise ValueError(f"task {task_id} is not '{frm}'")
            self.db.execute("update tasks set status=?, updated=? where id=?", (to, iso(self.t), task_id))
            pid, st = row["panel"], self.state[row["panel"]]
            if action == "approve":
                self._notify("staff", f"{pid}: new {row['kind']} task (approved by manager)")
            if action in ("reject", "done"):
                st["cooldown_until"] = self.t + timedelta(hours=self.settings["cooldown_hours"])
                st["streak"] = 0
            if action == "done":
                if row["kind"] == "cleaning":
                    self.twins[pid].clean()
                    st["last"] = st["overlay"] = None   # old dirty reading no longer describes the panel
                self._notify("manager", f"{pid}: {row['kind']} completed")
            self.db.commit()

    def update_settings(self, new):
        allowed = {"mode": str, "minutes_per_tick": int, "tick_seconds": float, "k": int,
                   "cooldown_hours": int, "price_tl_kwh": float}
        with self.lock:
            for key, val in new.items():
                if key not in allowed:
                    raise ValueError(f"unknown setting {key}")
                val = allowed[key](val)
                if key == "mode" and val not in ("manual", "auto"):
                    raise ValueError("mode must be manual or auto")
                if key != "mode" and val <= 0:
                    raise ValueError(f"{key} must be > 0")
                self.settings[key] = val

    def dirty_panel(self, pid, kind="dust"):
        """Demo helper: make one panel dusty, or drop a flock's worth of droppings on it, right now."""
        if kind not in ("dust", "birds"):
            raise ValueError("kind must be dust or birds")
        with self.lock:
            tw = self.twins[pid]
            if kind == "dust":
                tw.dust = 0.7
            else:
                for _ in range(12):
                    tw.step(0, drop=True)

    # ---- read side ------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            open_tasks = {r["panel"]: dict(r) for r in self.db.execute(
                f"select * from tasks where status in {OPEN} order by id")}
            panels = []
            for pid, st in self.state.items():
                last = st["last"] or {}
                panels.append({"id": pid, "label": last.get("label"), "reason": last.get("reason"),
                               "dirt": (last.get("features") or {}).get("dirt"), "power": st["power"],
                               "irr": st["irr"], "soiling": self.twins[pid].soiling, "streak": st["streak"],
                               "lost_kwh": st["lost_kwh"], "task": open_tasks.get(pid)})
            q = lambda sql, *a: [dict(r) for r in self.db.execute(sql, a)]  # noqa: E731
            return {"time": iso(self.t), "settings": dict(self.settings), "panels": panels,
                    "tasks": q("select * from tasks order by id desc limit 50"),
                    "notifications": {role: q("select * from notifications where role=? order by id desc limit 20", role)
                                      for role in ("manager", "staff")},
                    "unread": {role: self.db.execute("select count(*) from notifications where role=? and read=0",
                                                     (role,)).fetchone()[0] for role in ("manager", "staff")}}

    def mark_read(self, role):
        with self.lock:
            self.db.execute("update notifications set read=1 where role=?", (role,))
            self.db.commit()

    def history(self, pid, limit=200):
        with self.lock:
            rows = self.db.execute("select t,power,dirt,soiling,label from readings where panel=? "
                                   "order by id desc limit ?", (pid, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def image(self, pid, kind):
        with self.lock:
            return self.state[pid]["overlay" if kind == "overlay" else "frame"]


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def iso(t):
    return t.strftime("%Y-%m-%d %H:%M")
