"""Web layer: FastAPI serving the static pages in web/ and a small JSON API.

run: .venv/bin/uvicorn solarguard.web:app --port 8000
"""
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from .analyze import analyze, overlay
from .live import Live
from .panel import MAX_SIDE, load

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SOLARGUARD_DATA", ROOT / "data/raw/Faulty_solar_panel"))
EVAL = ROOT / "out/eval.json"
WEB = ROOT / "web"
N_PANELS = 6
MAX_UPLOAD = 10 * 1024 * 1024


def pick_bases(n=N_PANELS):
    """Clean photos the pipeline itself sees as clean, with a clearly found panel."""
    bases = []
    for f in sorted((DATA / "Clean").glob("*")):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = load(f)
        res, _ = analyze(img)
        f = res["features"]
        if res["label"] == "Temiz" and f["coverage"] > 0.7 and f["sat"] >= 90:
            bases.append(img)
            if len(bases) == n:
                break
    if not bases:
        raise RuntimeError(f"no usable clean images in {DATA}; run scripts/get_dataset.py")
    return bases


async def run_sim(live):
    while True:
        await asyncio.to_thread(live.tick)
        await asyncio.sleep(live.settings["tick_seconds"])


@asynccontextmanager
async def lifespan(app):
    # shortcut: in-memory DB and twin state reset on restart; persist both before a multi-day server demo.
    app.state.live = Live(pick_bases())
    task = asyncio.create_task(run_sim(app.state.live))
    yield
    task.cancel()


app = FastAPI(title="SolarGuard", lifespan=lifespan)


def live() -> Live:
    return app.state.live


def page(name):
    return lambda: FileResponse(WEB / name)


for route, name in {"/": "index.html", "/staff": "staff.html", "/dataset": "dataset.html", "/try": "try.html",
                    "/style.css": "style.css", "/common.js": "common.js"}.items():
    app.get(route, include_in_schema=False)(page(name))


def jpg_response(data):
    if data is None:
        raise HTTPException(404, "no image yet")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def check(fn):
    try:
        return fn()
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


# ---- Live Mode ---------------------------------------------------------------
@app.get("/api/state")
def state():
    return live().snapshot()


@app.get("/api/history/{pid}")
def history(pid: str):
    if pid not in live().twins:
        raise HTTPException(404, "unknown panel")
    return live().history(pid)


@app.get("/api/frame/{pid}")
def frame(pid: str, kind: str = "raw"):
    if pid not in live().twins:
        raise HTTPException(404, "unknown panel")
    return jpg_response(live().image(pid, kind))


@app.post("/api/tasks/{task_id}/{action}")
def task_action(task_id: int, action: str):
    check(lambda: live().act(task_id, action))
    return {"ok": True}


@app.post("/api/settings")
def settings(body: dict):
    check(lambda: live().update_settings(body))
    return live().settings


@app.post("/api/notifications/{role}/read")
def read_notifications(role: str):
    if role not in ("manager", "staff"):
        raise HTTPException(400, "unknown role")
    live().mark_read(role)
    return {"ok": True}


@app.post("/api/demo/dirty/{pid}")
def demo_dirty(pid: str, kind: str = "dust"):
    check(lambda: live().dirty_panel(pid, kind))
    return {"ok": True}


# ---- Dataset Mode ------------------------------------------------------------
_eval_cache = {}


def eval_data():
    if not EVAL.exists():
        raise HTTPException(404, "no evaluation yet: run scripts/eval_dataset.py")
    mtime = EVAL.stat().st_mtime
    if _eval_cache.get("mtime") != mtime:
        _eval_cache.update(mtime=mtime, data=json.loads(EVAL.read_text()))
    return _eval_cache["data"]


@app.get("/api/dataset")
def dataset():
    d = eval_data()
    return {k: d[k] for k in ("labels", "ms_per_image", "summary")} | {
        "images": [{k: r[k] for k in ("path", "kaggle", "split", "true", "pred", "reason")} for r in d["images"]]}


@app.get("/api/dataset/image/{idx}")
def dataset_image(idx: int, kind: str = "raw", size: int = MAX_SIDE):
    images = eval_data()["images"]
    if not 0 <= idx < len(images):
        raise HTTPException(404, "no such image")
    img = load(DATA / images[idx]["path"], max_side=min(max(size, 64), MAX_SIDE))
    if kind == "overlay":
        img = overlay(img, analyze(img)[1])
    return jpg_response(cv2.imencode(".jpg", img)[1].tobytes())


# ---- Try it (FR8) --------------------------------------------------------------
@app.post("/api/analyze")
async def analyze_upload(file: UploadFile):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image larger than 10 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res, masks = await asyncio.to_thread(analyze, img)
    b64 = lambda im: "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im)[1]).decode()  # noqa: E731
    return res | {"original": b64(img), "overlay": b64(overlay(img, masks)),
                  "dirt_mask": b64(masks["dirt"]), "panel_mask": b64(masks["panel"])}
