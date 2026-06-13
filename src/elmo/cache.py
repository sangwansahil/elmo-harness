"""SQLite-backed completion cache.

Keyed by sha256(provider + model + messages + temperature + max_tokens).
Identical calls within a project never bill twice. Disable via env
ELMO_NO_CACHE=1 or `--no-cache` on the cli.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elmo.providers.base import CompletionRequest


SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    finish_reason TEXT NOT NULL,
    created_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_completions_provider ON completions(provider, model);
"""


def cache_disabled() -> bool:
    return os.environ.get("ELMO_NO_CACHE", "").strip() in {"1", "true", "yes"}


class CompletionCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def make_key(provider: str, req: "CompletionRequest") -> str:
        payload = {
            "p": provider,
            "m": req.model,
            "msgs": req.messages,
            "t": round(req.temperature, 4),
            "mx": req.max_tokens,
            "stop": req.stop or None,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> dict | None:
        """Return the raw row dict (text, model, tokens, finish_reason) or None.

        Callers reconstruct domain types from this — keeps the cache
        decoupled from the provider module.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT response_text, model, prompt_tokens, completion_tokens, "
                "finish_reason FROM completions WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            c.execute("UPDATE completions SET hit_count = hit_count + 1 WHERE key = ?", (key,))
        return {
            "text": row["response_text"],
            "model": row["model"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "finish_reason": row["finish_reason"],
        }

    def put(
        self,
        key: str,
        provider: str,
        *,
        text: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO completions "
                "(key, provider, model, response_text, prompt_tokens, completion_tokens, "
                "finish_reason, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (key, provider, model, text, prompt_tokens, completion_tokens,
                 finish_reason, time.time()),
            )

    def stats(self) -> dict:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) as n, SUM(hit_count) as hits, "
                "SUM(prompt_tokens) as pt, SUM(completion_tokens) as ct FROM completions"
            ).fetchone()
        return {
            "entries": row["n"] or 0,
            "total_hits": row["hits"] or 0,
            "prompt_tokens_saved": (row["pt"] or 0) * (row["hits"] or 0),
            "completion_tokens_saved": (row["ct"] or 0) * (row["hits"] or 0),
        }

    def clear(self) -> int:
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
            c.execute("DELETE FROM completions")
        return n
