# tests/test_middlewares.py
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient
from app.middlewares import logging_middleware

def _build_app():
    app = FastAPI()
    app.middleware("http")(logging_middleware)
    @app.get("/ok")
    async def ok(): return {"ok": True}
    @app.get("/boom")
    async def boom(): raise HTTPException(status_code=418, detail="teapot")
    return app

def test_middleware_ok_flow():
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/ok")
        assert r.status_code == 200

def test_middleware_error_flow():
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/boom")
        assert r.status_code == 418
