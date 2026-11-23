import numpy as np
import pytest
from fastapi.testclient import TestClient

from solarguard import web
from solarguard.live import Live


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(web.PASSWORDS, "manager", "m-pass")
    monkeypatch.setitem(web.PASSWORDS, "staff", "s-pass")
    monkeypatch.setattr(web, "make_live", lambda: Live([np.zeros((60, 80, 3), np.uint8)]))
    with TestClient(web.app) as c:
        yield c


def login(c, user, password):
    return c.post("/api/login", json={"user": user, "password": password})


def suggested_task(c):
    lv = web.app.state.live
    with lv.lock:
        lv._task("P-01", "cleaning", "suggested", "test")
        lv.db.commit()
        return lv.db.execute("select max(id) from tasks").fetchone()[0]


def test_everything_needs_login(client):
    assert client.get("/api/state").status_code == 401
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert client.get("/login").status_code == 200


def test_wrong_password_rejected(client):
    assert login(client, "manager", "nope").status_code == 401
    assert login(client, "admin", "m-pass").status_code == 401


def test_tampered_cookie_rejected(client):
    login(client, "staff", "s-pass")
    forged = client.cookies.get(web.COOKIE).replace("staff", "manager", 1)
    client.cookies.set(web.COOKIE, forged)
    assert client.get("/api/me").status_code == 401


def test_staff_cannot_do_manager_things(client):
    tid = suggested_task(client)
    login(client, "staff", "s-pass")
    assert client.get("/api/me").json() == {"role": "staff"}
    assert client.get("/", follow_redirects=False).headers["location"] == "/staff"
    assert client.post(f"/api/tasks/{tid}/approve").status_code == 403
    assert client.post("/api/settings", json={"mode": "auto"}).status_code == 403
    assert client.post("/api/demo/dirty/P-01").status_code == 403
    assert client.post("/api/notifications/manager/read").status_code == 403


def test_manager_approves_staff_finishes(client):
    tid = suggested_task(client)
    login(client, "manager", "m-pass")
    assert client.post(f"/api/tasks/{tid}/approve").status_code == 200
    client.post("/api/logout")
    login(client, "staff", "s-pass")
    assert client.post(f"/api/tasks/{tid}/start").status_code == 200
    assert client.post(f"/api/tasks/{tid}/done").status_code == 200
