"""FastAPI app — REST + WebSocket + static UI."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from elmo.config import Paths


# WebSocket types must be importable from module scope so FastAPI's deferred-
# annotation resolution (under `from __future__ import annotations`) succeeds
# at route registration. Importing them inside create_app() leaves the route
# registered as plain WebSocket but the dependency resolver couldn't bind the
# parameter and Starlette rejected the upgrade with 403. We import lazily so
# missing fastapi (in non-ui installs) is still surfaced via _require_fastapi.
try:
    from fastapi import WebSocket, WebSocketDisconnect  # noqa: F401
except ImportError:  # pragma: no cover — surfaced by _require_fastapi later
    WebSocket = object  # type: ignore[assignment,misc]
    WebSocketDisconnect = Exception  # type: ignore[assignment,misc]


def asdict_state(state) -> dict:
    return asdict(state)


def asdict_entry(entry) -> dict:
    return asdict(entry)


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
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    paths = paths or Paths.from_cwd()
    paths.ensure()

    # Initialize the SQLite schema up-front so the wizard can hit /api/runs
    # before any `elmo run` has been executed. Storage() is idempotent.
    from elmo.storage import Storage
    Storage(paths.db)

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

    # --- onboarding wizard endpoints ----------------------------------------

    @app.get("/api/system")
    def system_probe():
        from elmo.system import probe
        return probe().asdict()

    @app.get("/api/catalog")
    def catalog(specialty: str | None = None):
        from elmo.catalog import recommend
        from elmo.system import probe
        p = probe()
        recs = recommend(p, specialty=specialty)  # type: ignore[arg-type]
        return {"system": p.asdict(), "models": [r.asdict() for r in recs]}

    @app.post("/api/models/{model_id}/download")
    def model_download(model_id: str):
        from elmo.catalog import CATALOG
        from elmo.download import start_download

        m = next((x for x in CATALOG if x.id == model_id), None)
        if not m:
            raise HTTPException(404, "unknown model id")
        st = start_download(
            hf_id=m.hf_id_mlx,
            display_name=m.display_name,
            bytes_estimate=int(m.disk_4bit_gb * (1024**3)),
        )
        return {"download_id": st.id, "state": dict(asdict_state(st))}

    @app.get("/api/downloads/{download_id}")
    def download_status(download_id: str):
        from elmo.download import get_state
        st = get_state(download_id)
        if not st:
            raise HTTPException(404, "no such download")
        return asdict_state(st)

    @app.websocket("/api/downloads/{download_id}/live")
    async def download_live(ws: WebSocket, download_id: str):
        from elmo.download import get_state
        await ws.accept()
        try:
            while True:
                st = get_state(download_id)
                if not st:
                    await ws.send_json({"error": "missing"})
                    break
                await ws.send_json(asdict_state(st))
                if st.status in ("done", "error", "cancelled"):
                    break
                await asyncio.sleep(0.6)
        except WebSocketDisconnect:
            return

    @app.post("/api/probe")
    def probe_model(payload: dict):
        """Smoke-test a downloaded base model with one prompt."""
        import traceback
        from elmo.backends import get_backend
        from elmo.config import detect_backend

        hf_id = payload.get("hf_id")
        prompt = (payload.get("prompt") or "say hi in one short sentence.").strip()
        if not hf_id:
            raise HTTPException(400, "hf_id required")
        backend_name = payload.get("backend") or detect_backend()
        if backend_name == "none":
            raise HTTPException(400, "no local backend available on this machine")
        try:
            backend = get_backend(backend_name)
            out = backend.generate(model_id=hf_id, adapter_path=None,
                                   prompts=[prompt], max_tokens=128)
        except Exception as e:
            tb = traceback.format_exc()
            # Surface the actual failure as JSON instead of a 500 Internal Server
            # Error html body the client can't parse. Truncate to keep payloads small.
            raise HTTPException(
                status_code=500,
                detail={
                    "error": type(e).__name__,
                    "message": str(e),
                    "traceback": tb.splitlines()[-12:],
                    "hint": "see `.elmo/daemon.log` for the full trace.",
                },
            ) from e
        return {"prompt": prompt, "completion": out[0] if out else "", "model": hf_id}

    @app.post("/api/downloads/{download_id}/cancel")
    def cancel_download(download_id: str):
        from elmo.download import cancel, get_state
        ok = cancel(download_id)
        st = get_state(download_id)
        return {"ok": ok, "state": asdict_state(st) if st else None}

    @app.get("/api/logs")
    def tail_logs(tail: int = 100):
        """Tail .elmo/daemon.log. Used by the ui to surface server-side
        traces inline next to client-side errors."""
        log_path = Path.cwd() / ".elmo" / "daemon.log"
        if not log_path.exists():
            return {"path": str(log_path), "lines": [], "exists": False}
        tail = max(1, min(tail, 500))
        # cheap tail: read last N lines without slurping the whole file.
        try:
            with log_path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = min(size, 64 * 1024)
                f.seek(size - block)
                buf = f.read().decode("utf-8", errors="replace")
            lines = buf.splitlines()[-tail:]
        except OSError as e:
            lines = [f"<read failed: {e}>"]
        return {"path": str(log_path), "lines": lines, "exists": True}

    @app.get("/api/hub")
    def hub_list(kind: str | None = None):
        from elmo.hub import list_models
        return {"entries": [
            {**asdict_entry(e)} for e in list_models(kind=kind)
        ]}

    @app.post("/api/hub/save")
    def hub_save(payload: dict):
        from elmo.hub import save_finetune
        missing = [k for k in ("display_name", "base_hf_id", "adapter_path",
                               "task_name", "run_id") if not payload.get(k)]
        if missing:
            raise HTTPException(400, f"missing fields: {missing}")
        entry = save_finetune(
            display_name=payload["display_name"],
            base_hf_id=payload["base_hf_id"],
            adapter_path=payload["adapter_path"],
            task_name=payload["task_name"],
            run_id=payload["run_id"],
            baseline_score=payload.get("baseline_score"),
            final_score=payload.get("final_score"),
        )
        return asdict_entry(entry)

    @app.delete("/api/hub/{entry_id}")
    def hub_delete(entry_id: str):
        from elmo.hub import remove
        ok = remove(entry_id)
        if not ok:
            raise HTTPException(404, "no such entry")
        return {"ok": True}

    # --- wizard task discovery + start --------------------------------------

    @app.post("/api/wizard/discover")
    def wizard_discover(payload: dict):
        """Turn (prompt, base_model_hf_id) into a previewable TaskSpec."""
        from elmo.wizard import classify_prompt, discover_task

        prompt = (payload.get("prompt") or "").strip()
        base = payload.get("base_model")
        if not prompt or not base:
            raise HTTPException(400, "prompt and base_model required")
        guess = classify_prompt(prompt)
        spec = discover_task(prompt, base)
        gates = []
        for cap in spec.capabilities:
            v = cap.verifier
            text = {
                "function_call": "function name and arguments must match the expected call",
                "json_schema": "output must validate against the json schema",
                "exact_match": "extracted answer must equal ground truth (numeric tolerance ok)",
                "exec_python": "generated python must execute and return the expected value",
                "judge": "an llm judge will score against the rubric",
            }.get(v, "rubric-checked by a judge")
            gates.append({"capability": cap.name, "verifier": v, "rule": text, "weight": cap.weight})

        # Surface dataset access status so the review step can hand-hold the
        # user through requesting HF access before they hit start training.
        access = None
        if spec.dataset.source.startswith("hf:"):
            from elmo.hf_auth import dataset_access
            repo_id = spec.dataset.source[3:]
            access = dataset_access(repo_id).asdict()

        return {
            "guess": {
                "kind": guess.kind,
                "confidence": guess.confidence,
                "matched_keywords": guess.matched_keywords,
            },
            "spec": spec.model_dump(),
            "gates": gates,
            "dataset_pretty": {
                "source": spec.dataset.source,
                "kind": "hugging face" if spec.dataset.source.startswith("hf:") else "synthetic",
                "max_rows": spec.dataset.max_rows,
            },
            "access": access,
        }

    @app.get("/api/hf/whoami")
    def hf_whoami():
        from elmo.hf_auth import whoami
        return whoami().asdict()

    @app.post("/api/hf/token")
    def hf_set_token(payload: dict):
        from elmo.hf_auth import save_token, whoami
        token = (payload.get("token") or "").strip()
        if not token:
            raise HTTPException(400, "token required")
        who = whoami(token)
        if not who.configured:
            raise HTTPException(400, who.error or "invalid token")
        save_token(token)
        return who.asdict()

    @app.get("/api/hf/dataset")
    def hf_dataset_access(repo_id: str):
        from elmo.hf_auth import dataset_access
        return dataset_access(repo_id).asdict()

    @app.post("/api/wizard/start")
    def wizard_start(payload: dict):
        """Take a confirmed spec dict and kick off the run loop in the daemon."""
        from elmo.server.runner import start_run
        from elmo.spec import TaskSpec

        spec_payload = payload.get("spec")
        if not isinstance(spec_payload, dict):
            raise HTTPException(400, "spec (dict) required")
        try:
            spec = TaskSpec.model_validate(spec_payload)
        except Exception as e:
            raise HTTPException(400, f"invalid spec: {e}") from e
        handle = start_run(spec, paths)
        return {
            "run_id": handle.run_id,
            "task_name": handle.task_name,
            "base_model": handle.base_model,
            "status": handle.status,
            "started_at": handle.started_at,
        }

    @app.get("/api/wizard/runs/{run_id}")
    def wizard_run_status(run_id: str):
        from elmo.server.runner import get_handle
        h = get_handle(run_id)
        if not h:
            raise HTTPException(404, "no such handle")
        return {
            "run_id": h.run_id,
            "task_name": h.task_name,
            "base_model": h.base_model,
            "status": h.status,
            "started_at": h.started_at,
            "error": h.error,
            "thread_alive": h.thread_alive,
            "extra": h.extra,
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
