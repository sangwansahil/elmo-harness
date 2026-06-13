"""Monotonically-growing regression suite.

Each failure ever observed becomes a permanent test case. The suite is stored
as an append-only JSONL on disk and tagged with the capability that broke,
the iteration it was first seen on, and (when applicable) the iteration that
fixed it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RegressionCase:
    id: str
    capability: str
    query: str
    tools: list
    expected_calls: list
    system: str
    first_seen_iter: int
    fixed_in_iter: int | None = None
    source_run_id: str = ""
    notes: str = ""

    def to_eval_row(self) -> dict:
        return {
            "query": self.query,
            "tools": self.tools,
            "expected_calls": self.expected_calls,
            "system": self.system,
        }


class RegressionSuite:
    """Append-only JSONL store. Idempotent on (capability, query) pairs."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")
        self._cases: list[RegressionCase] = self._load()

    def _load(self) -> list[RegressionCase]:
        cases: list[RegressionCase] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(RegressionCase(**obj))
        return cases

    def _key(self, capability: str, query: str) -> str:
        return f"{capability}::{query.strip()[:200]}"

    def add_failure(
        self,
        *,
        capability: str,
        query: str,
        tools: list,
        expected_calls: list,
        system: str,
        iteration: int,
        source_run_id: str,
        notes: str = "",
    ) -> RegressionCase | None:
        """Promote a failure into the suite. Returns the new case, or None if
        a matching (capability, query) is already tracked."""
        key = self._key(capability, query)
        for c in self._cases:
            if self._key(c.capability, c.query) == key:
                return None
        case = RegressionCase(
            id=f"rc_{uuid.uuid4().hex[:8]}",
            capability=capability,
            query=query,
            tools=tools,
            expected_calls=expected_calls,
            system=system,
            first_seen_iter=iteration,
            source_run_id=source_run_id,
            notes=notes,
        )
        self._cases.append(case)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(case)) + "\n")
        return case

    def mark_fixed(self, case_id: str, iteration: int) -> None:
        rewrite = False
        for c in self._cases:
            if c.id == case_id and c.fixed_in_iter is None:
                c.fixed_in_iter = iteration
                rewrite = True
        if rewrite:
            self._rewrite()

    def _rewrite(self) -> None:
        with self.path.open("w") as f:
            for c in self._cases:
                f.write(json.dumps(asdict(c)) + "\n")

    @property
    def cases(self) -> list[RegressionCase]:
        return list(self._cases)

    def by_capability(self) -> dict[str, list[RegressionCase]]:
        out: dict[str, list[RegressionCase]] = {}
        for c in self._cases:
            out.setdefault(c.capability, []).append(c)
        return out

    def write_eval_jsonl(self, path: Path, capability: str | None = None) -> int:
        """Materialize the suite as an eval jsonl that the FunctionCallEvaluator
        can consume."""
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w") as f:
            for c in self._cases:
                if capability is not None and c.capability != capability:
                    continue
                f.write(json.dumps(c.to_eval_row()) + "\n")
                n += 1
        return n
