"""Tests for the daemon — endpoints respond and serve the UI shell."""

from __future__ import annotations

from pathlib import Path

import pytest


def _client(tmp_path: Path):
    fastapi = pytest.importorskip("fastapi")
    starlette = pytest.importorskip("starlette.testclient")  # noqa: F841
    from fastapi.testclient import TestClient

    from elmo.config import Paths
    from elmo.server.app import create_app
    from elmo.storage import Storage

    paths = Paths(root=tmp_path / ".elmo", db=tmp_path / ".elmo" / "elmo.db",
                  runs=tmp_path / "runs", cache=tmp_path / ".elmo" / "cache")
    paths.ensure()
    storage = Storage(paths.db)
    storage.create_run("run_test_1", "function-calling", "{}", "qwen-1.5b", "mlx",
                       str(paths.runs / "run_test_1"))
    storage.update_run("run_test_1", baseline_score=0.5, final_score=0.7, status="done")
    storage.log_event("run_test_1", "data", "loaded xlam")
    storage.log_event("run_test_1", "train", "200 steps")
    storage.record_iteration("run_test_1", 0, 0.5, None, "baseline")
    storage.record_iteration("run_test_1", 1, 0.7, 0.2, "iter 1")

    app = create_app(paths)
    return TestClient(app)


def test_list_runs(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/api/runs")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == "run_test_1"
    assert rows[0]["final_score"] == 0.7


def test_run_detail_with_events(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/api/runs/run_test_1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "run_test_1"
    assert len(body["iterations"]) == 2

    e = c.get("/api/runs/run_test_1/events")
    assert e.status_code == 200
    events = e.json()
    assert len(events) == 2
    assert events[0]["stage"] == "data"


def test_run_detail_404(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/api/runs/nope")
    assert r.status_code == 404


def test_health_endpoint(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "backend" in body and "providers_configured" in body


def test_regression_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    c = _client(tmp_path)
    r = c.get("/api/regression/function-calling")
    assert r.status_code == 200
    assert r.json() == {"task": "function-calling", "cases": [], "by_capability": {}}


def test_ui_index_served(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "elmo" in r.text.lower()
    assert "prompt" in r.text.lower()
