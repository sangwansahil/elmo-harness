"""Background run executor — kicks off elmo.run.execute() in a thread so the
daemon HTTP loop stays responsive while training proceeds.

The thread writes events to the same SQLite the WebSocket already reads, so
no extra plumbing is needed for live updates.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from elmo.config import Paths


@dataclass
class RunHandle:
    run_id: str
    task_name: str
    base_model: str
    status: str               # "queued" | "running" | "done" | "error"
    started_at: str
    error: str = ""
    thread_alive: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


_handles: dict[str, RunHandle] = {}
_lock = threading.Lock()


def list_handles() -> list[RunHandle]:
    with _lock:
        return list(_handles.values())


def get_handle(run_id: str) -> RunHandle | None:
    with _lock:
        return _handles.get(run_id)


def start_run(spec, paths: Paths) -> RunHandle:
    """Kick off a run in a daemon thread. Returns immediately with a handle."""
    from elmo.run import _new_run_id  # internal but stable

    run_id = _new_run_id()
    handle = RunHandle(
        run_id=run_id,
        task_name=spec.name,
        base_model=spec.base_model,
        status="queued",
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    with _lock:
        _handles[run_id] = handle

    def _worker():
        from elmo.run import execute
        from elmo.storage import Storage

        try:
            with _lock:
                _handles[run_id].status = "running"
            # Pass run_id so the websocket / events all share one id.
            result = execute(spec, paths, run_id=run_id)
            with _lock:
                _handles[run_id].status = "done"
                _handles[run_id].extra = {
                    "actual_run_id": result.run_id,
                    "baseline_overall": result.baseline.overall,
                    "best_overall": result.best.overall,
                    "best_iteration": result.best_iteration,
                    "adapter_path": str(result.adapter_path) if result.adapter_path else "",
                    "artifact_dir": str(result.artifact_dir),
                }
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            try:
                Storage(paths.db).log_event(run_id, "error", f"{e!r}\n{tb[:1200]}", level="error")
            except Exception:
                pass
            with _lock:
                _handles[run_id].status = "error"
                _handles[run_id].error = repr(e)
        finally:
            with _lock:
                _handles[run_id].thread_alive = False

    t = threading.Thread(target=_worker, daemon=True, name=f"elmo-run-{run_id}")
    t.start()
    return handle
