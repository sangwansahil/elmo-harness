"""User-level config: where elmo lives on disk."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    db: Path
    runs: Path
    cache: Path

    @classmethod
    def from_cwd(cls, cwd: Path | None = None) -> "Paths":
        cwd = cwd or Path.cwd()
        root = cwd / ".elmo"
        return cls(
            root=root,
            db=root / "elmo.db",
            runs=cwd / "runs",
            cache=root / "cache",
        )

    def ensure(self) -> None:
        for p in (self.root, self.runs, self.cache):
            p.mkdir(parents=True, exist_ok=True)


def detect_backend() -> str:
    """Pick the default training backend for this machine."""
    if os.uname().sysname == "Darwin" and os.uname().machine == "arm64":
        return "mlx"
    try:
        import torch  # noqa: F401

        if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available():
            return "unsloth"
    except ImportError:
        pass
    return "none"
