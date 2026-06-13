"""FastAPI app — REST + WebSocket + static UI."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from elmo.config import Paths


UI_DIR = Path(__file__).parent / "ui"


def _require_fastapi() -> Any:
    try:
        import fastapi  # noqa: F401
        return fastapi
    except ImportError as e:
        raise RuntimeError(
            "fastapi not installed. install the ui extra: pip install 'elmo-harness[ui]'"
        ) from e


def create_app(paths: Paths | None = None):
    fastapi = _require_fastapi()
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    paths = paths or Paths.from_cwd()
    paths.ensure()

    app = FastAPI(
        title="elmo daemon",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    @contextmanager
    def _conn():
        c = sqlite3.connect(paths.db)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    @app.get("/api/runs")
    def list_runs(limit: int = 50):
        with _conn() as c:
            rows = c.execute(
                "SELECT id, task_name, base_model, status, baseline_score, "
                "final_score, created_at, updated_at FROM runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        with _conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                raise HTTPException(404, "run not found")
            iters = c.execute(
                "SELECT iter_n, score, delta, note, created_at FROM iterations "
                "WHERE run_id = ? ORDER BY iter_n",
                (run_id,),
            ).fetchall()
        run = dict(row)
        # parse spec for capability list + budget
        try:
            run["spec"] = json.loads(run.get("spec_json") or "{}")
        except json.JSONDecodeError:
            run["spec"] = {}
        run["iterations"] = [dict(r) for r in iters]
        # surface report.json if present
        report_path = Path(run["artifact_dir"]) / "report.json"
        if report_path.exists():
            try:
                run["report"] = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                run["report"] = None
        return run

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, since: float = 0, limit: int = 500):
        with _conn() as c:
            rows = c.execute(
                "SELECT id, iter_n, stage, level, message, data_json, created_at "
                "FROM events WHERE run_id = ? AND created_at > ? "
                "ORDER BY created_at LIMIT ?",
                (run_id, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    @app.websocket("/api/runs/{run_id}/live")
    async def run_live(ws: WebSocket, run_id: str):
        await ws.accept()
        last_seen = 0.0
        try:
            while True:
                with _conn() as c:
                    new = c.execute(
                        "SELECT id, iter_n, stage, level, message, data_json, created_at "
                        "FROM events WHERE run_id = ? AND created_at > ? "
                        "ORDER BY created_at",
                        (run_id, last_seen),
                    ).fetchall()
                    if new:
                        for r in new:
                            await ws.send_json(dict(r))
                            last_seen = max(last_seen, float(r["created_at"]))
                    status_row = c.execute(
                        "SELECT status FROM runs WHERE id = ?", (run_id,)
                    ).fetchone()
                if status_row and status_row["status"] in ("done", "failed"):
                    await ws.send_json({"_status": status_row["status"]})
                    break
                await asyncio.sleep(0.6)
        except WebSocketDisconnect:
            return

    @app.get("/api/regression/{task}")
    def regression(task: str):
        path = Path.cwd() / "runs" / f"{task}.regression.jsonl"
        if not path.exists():
            return {"task": task, "cases": [], "by_capability": {}}
        cases = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        by_cap: dict[str, list] = {}
        for c in cases:
            by_cap.setdefault(c["capability"], []).append(c)
        return {"task": task, "cases": cases, "by_capability": by_cap}

    @app.get("/api/health")
    def health():
        from elmo.config import detect_backend
        from elmo.providers import list_available
        return {
            "ok": True,
            "backend": detect_backend(),
            "providers_configured": list_available(),
        }

    # --- static UI ----------------------------------------------------------
    if (UI_DIR / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return FileResponse(UI_DIR / "index.html")

        @app.get("/style.css")
        def style():
            return FileResponse(UI_DIR / "style.css", media_type="text/css")

        @app.get("/app.js")
        def appjs():
            return FileResponse(UI_DIR / "app.js", media_type="application/javascript")

    return app


def serve(host: str = "127.0.0.1", port: int = 7777, reload: bool = False) -> None:
    _require_fastapi()
    import uvicorn

    uvicorn.run(
        "elmo.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
