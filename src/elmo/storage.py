"""SQLite run state. Plain stdlib, no ORM."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    base_model TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    baseline_score REAL,
    final_score REAL,
    artifact_dir TEXT
);

CREATE TABLE IF NOT EXISTS iterations (
    run_id TEXT NOT NULL,
    iter_n INTEGER NOT NULL,
    score REAL,
    delta REAL,
    note TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, iter_n),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iter_n INTEGER,
    stage TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    data_json TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, created_at);
"""


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(
        self,
        run_id: str,
        task_name: str,
        spec_json: str,
        base_model: str,
        backend: str,
        artifact_dir: str,
    ) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, task_name, spec_json, base_model, backend, status, "
                "created_at, updated_at, artifact_dir) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, task_name, spec_json, base_model, backend, "running",
                 now, now, artifact_dir),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))

    def log_event(
        self,
        run_id: str,
        stage: str,
        message: str,
        level: str = "info",
        iter_n: int | None = None,
        data: dict | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO events (run_id, iter_n, stage, level, message, data_json, "
                "created_at) VALUES (?,?,?,?,?,?,?)",
                (run_id, iter_n, stage, level, message,
                 json.dumps(data) if data else None, time.time()),
            )

    def record_iteration(
        self,
        run_id: str,
        iter_n: int,
        score: float,
        delta: float | None,
        note: str = "",
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO iterations (run_id, iter_n, score, delta, note, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (run_id, iter_n, score, delta, note, time.time()),
            )

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, task_name, base_model, status, baseline_score, final_score, "
                "created_at FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
