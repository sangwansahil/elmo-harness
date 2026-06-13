"""Trajectory artifacts — the open prior that lets every run start warm.

A Trajectory is the structured outcome of one run:
  (task_spec, base_model, data_recipe, train_config, eval_deltas, regression_size)

They are written locally as JSONL and can be uploaded to a public Hugging Face
dataset so the planner has a growing corpus of "what worked elsewhere" to
condition on. Closed-source harnesses cannot compound across customers; an
open one can — this is the structural moat.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


_TOK_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Trajectory:
    id: str
    task_name: str
    prompt: str
    base_model: str
    backend: str
    capabilities: list[str]
    objective: str
    dataset_source: str
    foundry_enabled: bool
    n_iterations: int
    baseline_overall: float
    best_overall: float
    delta: float
    per_capability_delta: dict[str, float] = field(default_factory=dict)
    regression_suite_size: int = 0
    planner_model: str = ""
    generator_model: str = ""
    created_at: float = field(default_factory=time.time)

    def tokens(self) -> set[str]:
        bag = " ".join([
            self.task_name, self.prompt, self.base_model,
            self.objective, self.dataset_source, " ".join(self.capabilities),
        ]).lower()
        return set(_TOK_RE.findall(bag))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class TrajectoryStore:
    """Append-only JSONL of trajectories. Local first; sync to HF on demand."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")

    def add(self, t: Trajectory) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(t)) + "\n")

    def all(self) -> list[Trajectory]:
        out: list[Trajectory] = []
        for ln in self.path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            out.append(Trajectory(**obj))
        return out

    def search(self, query: Trajectory, k: int = 3) -> list[tuple[float, Trajectory]]:
        qtok = query.tokens()
        scored = [(jaccard(qtok, t.tokens()), t) for t in self.all() if t.id != query.id]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


def trajectory_from_report(report: dict, spec: dict) -> Trajectory:
    """Build a Trajectory from a finished run's report.json + spec.

    Both args are JSON-ish dicts so this function has no pydantic dependency
    and can be used outside the run loop.
    """
    baseline = report.get("baseline", {})
    best = report.get("best", {})
    per_cap = {}
    for k in ("tool_selection", "arguments", "parallel_calls", "overall"):
        if k in best and k in baseline:
            per_cap[k] = round(float(best[k]) - float(baseline[k]), 4)
    base_overall = float(baseline.get("overall", 0.0))
    best_overall = float(best.get("overall", base_overall))
    tid_seed = f"{spec.get('name', '?')}|{spec.get('base_model', '?')}|{time.time()}"
    return Trajectory(
        id="t_" + hashlib.sha256(tid_seed.encode()).hexdigest()[:12],
        task_name=spec.get("name", "?"),
        prompt=spec.get("prompt", ""),
        base_model=spec.get("base_model", "?"),
        backend=spec.get("backend", "auto"),
        capabilities=[c["name"] for c in spec.get("capabilities", []) if "name" in c],
        objective=spec.get("train", {}).get("objective", "sft"),
        dataset_source=spec.get("dataset", {}).get("source", ""),
        foundry_enabled=bool(spec.get("foundry", {}).get("enabled")),
        n_iterations=len(report.get("iterations", [])),
        baseline_overall=base_overall,
        best_overall=best_overall,
        delta=round(best_overall - base_overall, 4),
        per_capability_delta=per_cap,
        regression_suite_size=int(report.get("regression_suite", {}).get("total_cases", 0)),
    )


def publish_to_hf(
    trajectories: list[Trajectory],
    repo_id: str,
    file_name: str = "trajectories.jsonl",
    token: str | None = None,
) -> str:
    """Upload to a Hugging Face dataset repo. Requires HF_TOKEN env or `token`."""
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. install with: pip install huggingface_hub"
        ) from e
    tok = token or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN env var (or `token=`) is required to publish.")
    payload = "\n".join(json.dumps(asdict(t)) for t in trajectories) + "\n"
    tmp = Path("/tmp") / f"elmo-{file_name}"
    tmp.write_text(payload)
    api = HfApi(token=tok)
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"add {len(trajectories)} trajectories",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def fetch_from_hf(
    repo_id: str,
    file_name: str = "trajectories.jsonl",
    token: str | None = None,
) -> list[Trajectory]:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed.") from e
    tok = token or os.environ.get("HF_TOKEN")
    local = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset", token=tok)
    out: list[Trajectory] = []
    for ln in Path(local).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(Trajectory(**json.loads(ln)))
    return out
