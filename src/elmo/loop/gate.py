"""Per-capability deployment gate — the vector compare.

A new iteration may deploy iff every tracked capability either improves or
stays equal versus the prior best. Aggregate-only comparison is rejected
because it hides regressions inside a higher overall number.
"""

from __future__ import annotations

from dataclasses import dataclass

from elmo.eval.function_calling import ScoreReport


@dataclass
class GateResult:
    passed: bool
    deltas: dict[str, float]
    regressions: list[str]
    reason: str = ""


def capability_vector(report: ScoreReport) -> dict[str, float]:
    return {
        "tool_selection": float(report.tool_selection),
        "arguments": float(report.arguments),
        "parallel_calls": float(report.parallel_calls),
        "overall": float(report.overall),
    }


def evaluate_gate(
    new: ScoreReport,
    baseline: ScoreReport,
    *,
    epsilon: float = 0.005,
    gated_capabilities: list[str] | None = None,
) -> GateResult:
    a = capability_vector(new)
    b = capability_vector(baseline)
    gated = set(gated_capabilities or ["tool_selection", "arguments", "parallel_calls"])
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for k in a:
        delta = a[k] - b[k]
        deltas[k] = round(delta, 4)
        if k in gated and delta < -epsilon:
            regressions.append(k)
    if regressions:
        return GateResult(
            passed=False,
            deltas=deltas,
            regressions=regressions,
            reason=f"regressed on: {', '.join(regressions)}",
        )
    return GateResult(passed=True, deltas=deltas, regressions=[], reason="ok")
