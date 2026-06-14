"""Local model hub — base models the user has downloaded plus their
fine-tuned variants.

Stored as a JSONL at ~/.elmo-hub/hub.jsonl with one entry per model. Each
entry points at the on-disk path (or HF cache location) for the actual
weights. Idempotent on (kind, identifier).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HubEntry:
    id: str
    kind: str                    # "base" | "fine-tuned"
    display_name: str
    hf_id: str = ""              # provenance: where it came from
    path: str = ""               # local path to weights (folder)
    adapter_path: str = ""       # for fine-tuned: path to the LoRA adapter
    base_id: str = ""            # for fine-tuned: hub id of the base
    task_name: str = ""          # for fine-tuned: which task spec produced it
    run_id: str = ""             # for fine-tuned: source run
    baseline_score: float | None = None
    final_score: float | None = None
    size_gb: float = 0.0
    added_at: float = field(default_factory=time.time)


def hub_root() -> Path:
    root = Path(os.environ.get("ELMO_HUB_ROOT", str(Path.home() / ".elmo-hub")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def hub_jsonl() -> Path:
    return hub_root() / "hub.jsonl"


def _read_all() -> list[HubEntry]:
    p = hub_jsonl()
    if not p.exists():
        return []
    out: list[HubEntry] = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        out.append(HubEntry(**obj))
    return out


def _write_all(entries: list[HubEntry]) -> None:
    p = hub_jsonl()
    p.write_text("".join(json.dumps(asdict(e)) + "\n" for e in entries))


def list_models(kind: str | None = None) -> list[HubEntry]:
    entries = _read_all()
    if kind:
        entries = [e for e in entries if e.kind == kind]
    entries.sort(key=lambda e: e.added_at, reverse=True)
    return entries


def get(entry_id: str) -> HubEntry | None:
    for e in _read_all():
        if e.id == entry_id:
            return e
    return None


def _dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / (1024**3), 2)


def register_base(hf_id: str, display_name: str, path: Path | str) -> HubEntry:
    """Idempotent on (kind=base, hf_id)."""
    existing = _read_all()
    for e in existing:
        if e.kind == "base" and e.hf_id == hf_id:
            return e
    entry = HubEntry(
        id="hb_" + uuid.uuid4().hex[:10],
        kind="base",
        display_name=display_name,
        hf_id=hf_id,
        path=str(path),
        size_gb=_dir_size_gb(Path(path)),
    )
    existing.append(entry)
    _write_all(existing)
    return entry


def save_finetune(
    *,
    display_name: str,
    base_hf_id: str,
    adapter_path: Path | str,
    task_name: str,
    run_id: str,
    baseline_score: float | None = None,
    final_score: float | None = None,
) -> HubEntry:
    """Promote a finished run into the hub as a fine-tuned model."""
    existing = _read_all()
    base = next((e for e in existing if e.kind == "base" and e.hf_id == base_hf_id), None)
    entry = HubEntry(
        id="hf_" + uuid.uuid4().hex[:10],
        kind="fine-tuned",
        display_name=display_name,
        hf_id=base_hf_id,
        path=str(base.path) if base else "",
        adapter_path=str(adapter_path),
        base_id=base.id if base else "",
        task_name=task_name,
        run_id=run_id,
        baseline_score=baseline_score,
        final_score=final_score,
        size_gb=_dir_size_gb(Path(adapter_path)),
    )
    existing.append(entry)
    _write_all(existing)
    return entry


def remove(entry_id: str) -> bool:
    entries = _read_all()
    n_before = len(entries)
    entries = [e for e in entries if e.id != entry_id]
    if len(entries) < n_before:
        _write_all(entries)
        return True
    return False
