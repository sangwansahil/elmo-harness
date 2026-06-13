"""Tests for the closed loop — gate logic, regression suite, diagnose JSON parse."""

from __future__ import annotations

from pathlib import Path

from elmo.eval.function_calling import ScoreReport
from elmo.loop.gate import capability_vector, evaluate_gate
from elmo.loop.regression import RegressionSuite


def _report(c1: float, c2: float, parallel: float, overall: float) -> ScoreReport:
    return ScoreReport(
        n=10, tool_selection=c1, arguments=c2, parallel_calls=parallel,
        overall=overall, per_example=[],
    )


def test_gate_passes_when_all_capabilities_improve() -> None:
    base = _report(0.50, 0.40, 0.30, 0.40)
    new = _report(0.60, 0.50, 0.40, 0.50)
    result = evaluate_gate(new, base)
    assert result.passed
    assert result.regressions == []
    assert all(d >= 0 for d in result.deltas.values())


def test_gate_blocks_regression_in_one_capability() -> None:
    base = _report(0.80, 0.40, 0.30, 0.50)
    new = _report(0.50, 0.60, 0.40, 0.60)  # arguments up, tool_selection regressed
    result = evaluate_gate(new, base)
    assert not result.passed
    assert "tool_selection" in result.regressions


def test_gate_tolerates_epsilon_dip() -> None:
    base = _report(0.501, 0.40, 0.30, 0.40)
    new = _report(0.500, 0.50, 0.40, 0.50)  # tool_selection -0.001 (within epsilon)
    result = evaluate_gate(new, base, epsilon=0.01)
    assert result.passed


def test_gate_ignores_non_gated_capabilities() -> None:
    base = _report(0.50, 0.40, 0.80, 0.50)
    new = _report(0.60, 0.50, 0.60, 0.55)  # parallel regressed but excluded
    result = evaluate_gate(new, base, gated_capabilities=["tool_selection", "arguments"])
    assert result.passed


def test_capability_vector_shape() -> None:
    v = capability_vector(_report(0.1, 0.2, 0.3, 0.25))
    assert set(v) == {"tool_selection", "arguments", "parallel_calls", "overall"}


def test_regression_suite_idempotent_on_duplicate_query(tmp_path: Path) -> None:
    suite = RegressionSuite(tmp_path / "r.jsonl")
    added1 = suite.add_failure(
        capability="tool_selection", query="weather in Tokyo?",
        tools=[{"name": "w"}], expected_calls=[{"name": "w", "arguments": {"c": "Tokyo"}}],
        system="sys", iteration=1, source_run_id="run_a",
    )
    added2 = suite.add_failure(
        capability="tool_selection", query="weather in Tokyo?",
        tools=[{"name": "w"}], expected_calls=[{"name": "w", "arguments": {"c": "Tokyo"}}],
        system="sys", iteration=2, source_run_id="run_b",
    )
    assert added1 is not None and added2 is None
    assert len(suite.cases) == 1


def test_regression_suite_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    s1 = RegressionSuite(path)
    s1.add_failure(
        capability="arguments", query="convert 10 USD to EUR",
        tools=[{"name": "convert"}],
        expected_calls=[{"name": "convert", "arguments": {"amount": 10, "from": "USD", "to": "EUR"}}],
        system="sys", iteration=1, source_run_id="run_a",
    )
    s2 = RegressionSuite(path)
    assert len(s2.cases) == 1
    assert s2.cases[0].capability == "arguments"


def test_regression_suite_mark_fixed(tmp_path: Path) -> None:
    suite = RegressionSuite(tmp_path / "r.jsonl")
    case = suite.add_failure(
        capability="arguments", query="q", tools=[{"name": "a"}],
        expected_calls=[{"name": "a", "arguments": {}}], system="sys",
        iteration=1, source_run_id="run",
    )
    assert case is not None
    suite.mark_fixed(case.id, iteration=3)
    # Reload from disk to check persistence
    s2 = RegressionSuite(tmp_path / "r.jsonl")
    assert s2.cases[0].fixed_in_iter == 3


def test_diagnose_json_extraction() -> None:
    from elmo.loop.diagnose import _extract_json_object

    text = '```json\n{"summary": "args mismatch", "corrective_brief": "more nested args"}\n```'
    obj = _extract_json_object(text)
    assert obj["summary"] == "args mismatch"
    assert "nested" in obj["corrective_brief"]
