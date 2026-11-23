"""Web layer: FastAPI serving the static pages in web/ and a small JSON API.

run: .venv/bin/uvicorn solarguard.web:app --port 8000

Environment:
  SOLARGUARD_MANAGER_PASSWORD / SOLARGUARD_STAFF_PASSWORD  login passwords (random + printed if unset)
  SOLARGUARD_SECRET         cookie signing key (random if unset -> everyone is logged out on restart)
  SOLARGUARD_SECURE_COOKIE  set to 1 when served over HTTPS
  SOLARGUARD_DB             SQLite file (default out/solarguard.db)
  SOLARGUARD_DATA           Kaggle image folder
"""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from .analyze import analyze, overlay
from .live import Live, saved_state
from .panel import MAX_SIDE, load

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SOLARGUARD_DATA", ROOT / "data/raw/Faulty_solar_panel"))
DB_PATH = Path(os.environ.get("SOLARGUARD_DB", ROOT / "out/solarguard.db"))
EVAL = ROOT / "out/eval.json"
WEB = ROOT / "web"
N_PANELS = 6
MAX_UPLOAD = 10 * 1024 * 1024
log = logging.getLogger("uvicorn.error")

# ---- auth ------------------------------------------------------------------------
ROLES = ("manager", "staff")
HOME = {"manager": "/", "staff": "/staff"}
COOKIE = "sg_session"
SESSION_SECONDS = 12 * 3600
SECRET = (os.environ.get("SOLARGUARD_SECRET") or secrets.token_hex(32)).encode()
SECURE_COOKIE = os.environ.get("SOLARGUARD_SECURE_COOKIE") == "1"
PASSWORDS = {}
for _role in ROLES:
    PASSWORDS[_role] = os.environ.get(f"SOLARGUARD_{_role.upper()}_PASSWORD")
    if not PASSWORDS[_role]:
        PASSWORDS[_role] = secrets.token_urlsafe(9)
        print(f"[SolarGuard] no SOLARGUARD_{_role.upper()}_PASSWORD set; {_role} password for this run: "
              f"{PASSWORDS[_role]}", flush=True)


def sign(role, expires):
    msg = f"{role}|{expires}"
    return f"{msg}|{hmac.new(SECRET, msg.encode(), hashlib.sha256).hexdigest()}"


def current_role(request: Request):
    try:
        role, expires, mac = request.cookies.get(COOKIE, "").split("|")
    except ValueError:
        return None
    good = hmac.compare_digest(sign(role, expires).rsplit("|", 1)[1], mac)
    return role if good and role in ROLES and expires.isdigit() and int(expires) > time.time() else None


def need(*roles):
    def dep(request: Request):
        role = current_role(request)
        if not role:
            raise HTTPException(401, "please log in")
        if role not in roles:
            raise HTTPException(403, f"{role} cannot do this")
        return role
    return Depends(dep)


ANY = need(*ROLES)
MANAGER = need("manager")


# ---- startup -----------------------------------------------------------------------
def pick_bases(n=N_PANELS):
    """Clean photos the pipeline itself sees as clean, with a clearly found panel. Returns (ids, images)."""
    ids, bases = [], []
    for f in sorted((DATA / "Clean").glob("*")):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = load(f)
        res, _ = analyze(img)
        feat = res["features"]
        if res["label"] == "Temiz" and feat["coverage"] > 0.7 and feat["sat"] >= 90:
            ids.append(str(f.relative_to(DATA)))
            bases.append(img)
            if len(bases) == n:
                break
    if not bases:
        raise RuntimeError(f"no usable clean images in {DATA}; run scripts/get_dataset.py")
    return ids, bases


def make_live():
    """Continue the saved simulation if there is one, else start fresh."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = saved_state(str(DB_PATH))
    try:
        ids = data["base_ids"]
        bases = [load(DATA / i) for i in ids]
    except (TypeError, KeyError, ValueError):     # no save, or the saved photos are gone
        ids, bases = pick_bases()
    lv = Live(bases, str(DB_PATH), base_ids=ids)
    if data:
        lv.restore(data)
        log.info("SolarGuard: resumed simulation at %s", lv.t)
    return lv


async def run_sim(live):
    while True:
        try:
            await asyncio.to_thread(live.tick)
        except Exception:                          # keep the simulation alive; the error is in the log
            log.exception("simulation tick failed")
        await asyncio.sleep(live.settings["tick_seconds"])


@asynccontextmanager
async def lifespan(app):
    app.state.live = make_live()
    task = asyncio.create_task(run_sim(app.state.live))
    yield
    task.cancel()


app = FastAPI(title="SolarGuard", lifespan=lifespan)


def live() -> Live:
    return app.state.live


# ---- pages ---------------------------------------------------------------------------
def page(name, roles):
    def handler(request: Request):
        role = current_role(request)
        if not role:
            return RedirectResponse(f"/login?next={request.url.path}", 303)
        if role not in roles:
            return RedirectResponse(HOME[role], 303)
        # no-store: the browser must re-check login on every visit instead of showing a cached page
        return FileResponse(WEB / name, headers={"Cache-Control": "no-store"})
    return handler


for route, name, roles in [("/", "index.html", ("manager",)), ("/staff", "staff.html", ROLES),
                           ("/dataset", "dataset.html", ROLES), ("/try", "try.html", ROLES)]:
    app.get(route, include_in_schema=False)(page(name, roles))
for name in ("login.html", "style.css", "common.js"):
    app.get("/" + name.removesuffix(".html"), include_in_schema=False)(lambda name=name: FileResponse(WEB / name))


@app.post("/api/login")
def login(body: dict):
    role, password = str(body.get("user", "")), str(body.get("password", ""))
    if role not in ROLES or not hmac.compare_digest(password.encode(), PASSWORDS[role].encode()):
        time.sleep(1)   # shortcut: slows guessing per request only; add per-IP lockout if exposed widely
        raise HTTPException(401, "wrong user or password")
    expires = int(time.time()) + SESSION_SECONDS
    res = JSONResponse({"role": role, "home": HOME[role]})
    res.set_cookie(COOKIE, sign(role, expires), max_age=SESSION_SECONDS, httponly=True, samesite="lax",
                   secure=SECURE_COOKIE)
    return res


@app.post("/api/logout")
def logout():
    res = JSONResponse({"ok": True})
    res.delete_cookie(COOKIE)
    return res


@app.get("/api/me")
def me(role: str = ANY):
    return {"role": role}


def jpg_response(data):
    if data is None:
        raise HTTPException(404, "no image yet")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def check(fn):
    try:
        return fn()
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


# ---- Live Mode ---------------------------------------------------------------------
@app.get("/api/state")
def state(role: str = ANY):
    return live().snapshot()


@app.get("/api/history/{pid}")
def history(pid: str, role: str = ANY):
    if pid not in live().twins:
        raise HTTPException(404, "unknown panel")
    return live().history(pid)


@app.get("/api/frame/{pid}")
def frame(pid: str, kind: str = "raw", role: str = ANY):
    if pid not in live().twins:
        raise HTTPException(404, "unknown panel")
    return jpg_response(live().image(pid, kind))


TASK_ROLES = {"approve": ("manager",), "reject": ("manager",), "start": ROLES, "done": ROLES}


@app.post("/api/tasks/{task_id}/{action}")
def task_action(task_id: int, action: str, role: str = ANY):
    if role not in TASK_ROLES.get(action, ROLES):
        raise HTTPException(403, f"only the manager can {action}")
    check(lambda: live().act(task_id, action))
    return {"ok": True}


@app.post("/api/settings")
def settings(body: dict, role: str = MANAGER):
    check(lambda: live().update_settings(body))
    return live().settings


@app.post("/api/notifications/{who}/read")
def read_notifications(who: str, role: str = ANY):
    if who not in ROLES:
        raise HTTPException(400, "unknown role")
    if who != role and role != "manager":
        raise HTTPException(403, "not your notifications")
    live().mark_read(who)
    return {"ok": True}


@app.post("/api/demo/dirty/{pid}")
def demo_dirty(pid: str, kind: str = "dust", role: str = MANAGER):
    check(lambda: live().dirty_panel(pid, kind))
    return {"ok": True}


# ---- Dataset Mode --------------------------------------------------------------------
_eval_cache = {}


def eval_data():
    if not EVAL.exists():
        raise HTTPException(404, "no evaluation yet: run scripts/eval_dataset.py")
    mtime = EVAL.stat().st_mtime
    if _eval_cache.get("mtime") != mtime:
        _eval_cache.update(mtime=mtime, data=json.loads(EVAL.read_text()))
    return _eval_cache["data"]


@app.get("/api/dataset")
def dataset(role: str = ANY):
    d = eval_data()
    return {k: d[k] for k in ("labels", "ms_per_image", "summary")} | {
        "images": [{k: r[k] for k in ("path", "kaggle", "split", "true", "pred", "reason")} for r in d["images"]]}


@app.get("/api/dataset/image/{idx}")
def dataset_image(idx: int, kind: str = "raw", size: int = MAX_SIDE, role: str = ANY):
    images = eval_data()["images"]
    if not 0 <= idx < len(images):
        raise HTTPException(404, "no such image")
    img = load(DATA / images[idx]["path"], max_side=min(max(size, 64), MAX_SIDE))
    if kind == "overlay":
        img = overlay(img, analyze(img)[1])
    return jpg_response(cv2.imencode(".jpg", img)[1].tobytes())


# ---- Try it (FR8) ----------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze_upload(file: UploadFile, role: str = ANY):
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
