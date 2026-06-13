"""Tests for the trajectory store — tokens, jaccard, store round-trip, search."""

from __future__ import annotations

from pathlib import Path

from elmo.trajectory import Trajectory, TrajectoryStore, jaccard, trajectory_from_report


def _traj(name: str, base: str, caps: list[str], delta: float = 0.1) -> Trajectory:
    return Trajectory(
        id=f"t_{name}",
        task_name=name,
        prompt=f"build a {name} expert",
        base_model=base,
        backend="mlx",
        capabilities=caps,
        objective="sft",
        dataset_source="hf:test",
        foundry_enabled=True,
        n_iterations=2,
        baseline_overall=0.4,
        best_overall=0.4 + delta,
        delta=delta,
    )


def test_jaccard_basic() -> None:
    assert jaccard(set("ab"), set("ab")) == 1.0
    assert jaccard(set("ab"), set("cd")) == 0.0
    assert 0.0 < jaccard(set("abc"), set("cde")) < 1.0


def test_trajectory_tokens_include_capabilities() -> None:
    t = _traj("function-calling", "qwen", ["tool_selection", "arguments"])
    tok = t.tokens()
    assert "function" in tok
    assert "tool_selection" in tok or "tool" in tok
    assert "qwen" in tok


def test_store_append_and_read(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.add(_traj("a", "qwen", ["c1"]))
    store.add(_traj("b", "llama", ["c2"]))
    all_ = store.all()
    assert len(all_) == 2
    assert {t.task_name for t in all_} == {"a", "b"}


def test_search_ranks_by_overlap(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.add(_traj("function-calling-math", "qwen2.5", ["tool_selection", "arguments"]))
    store.add(_traj("summarization", "llama-3.2", ["rouge", "length"]))
    store.add(_traj("function-calling-finance", "qwen2.5", ["tool_selection", "parallel_calls"]))

    query = _traj("function-calling", "qwen2.5", ["tool_selection", "arguments"])
    hits = store.search(query, k=3)
    # the unrelated one should rank last
    assert hits[-1][1].task_name == "summarization"


def test_search_excludes_self(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "t.jsonl")
    t = _traj("a", "qwen", ["c1"])
    store.add(t)
    store.add(_traj("b", "qwen", ["c1"]))
    hits = store.search(t, k=5)
    assert all(h[1].id != t.id for h in hits)


def test_trajectory_from_report() -> None:
    spec = {
        "name": "fc",
        "prompt": "build me a fc expert",
        "base_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "backend": "mlx",
        "capabilities": [
            {"name": "tool_selection", "weight": 1},
            {"name": "arguments", "weight": 1.5},
        ],
        "dataset": {"source": "hf:Salesforce/xlam-function-calling-60k"},
        "train": {"objective": "rft"},
        "foundry": {"enabled": True},
    }
    report = {
        "baseline": {"overall": 0.50, "tool_selection": 0.7, "arguments": 0.4, "parallel_calls": 0.3},
        "best": {"overall": 0.70, "tool_selection": 0.9, "arguments": 0.6, "parallel_calls": 0.5},
        "iterations": [{"n": 1}, {"n": 2}, {"n": 3}],
        "regression_suite": {"total_cases": 14},
    }
    t = trajectory_from_report(report, spec)
    assert t.task_name == "fc"
    assert t.objective == "rft"
    assert t.n_iterations == 3
    assert t.delta == 0.20
    assert t.per_capability_delta["arguments"] == 0.20
    assert t.regression_suite_size == 14
    assert t.foundry_enabled is True
    assert t.capabilities == ["tool_selection", "arguments"]
