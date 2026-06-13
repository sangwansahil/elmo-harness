"""Verifier-as-reward — turn the function-calling checker into a scalar signal
RL trainers can consume.

Reward shape (per completion):
  0.00  no tool call parsed
  0.30  parsed but wrong function name
  0.65  right function, wrong args
  1.00  full match (name + args)

For parallel-call cases the reward is the mean per expected call, minus a
penalty if extra spurious calls appear.
"""

from __future__ import annotations

from elmo.eval.verifier import compare_calls, parse_tool_calls


def function_call_reward(completion: str, expected_calls: list[dict]) -> float:
    """Score one model completion against an expected-calls list."""
    predicted = parse_tool_calls(completion)
    if not predicted and expected_calls:
        return 0.0
    if not expected_calls and not predicted:
        return 1.0  # correct refusal
    verdicts = compare_calls(predicted, expected_calls)
    expected_v = verdicts[: len(expected_calls)]
    per: list[float] = []
    for v in expected_v:
        if v.name_match and v.args_match:
            per.append(1.0)
        elif v.name_match:
            per.append(0.65)
        else:
            per.append(0.30 if predicted else 0.0)
    extras = sum(1 for v in verdicts if v.extra)
    base = sum(per) / max(1, len(expected_v))
    return max(0.0, base - 0.15 * extras)


def batch_function_call_rewards(
    completions: list[str], expected_calls: list[list[dict]]
) -> list[float]:
    """One reward per completion. Lengths must match."""
    if len(completions) != len(expected_calls):
        raise ValueError(
            f"length mismatch: {len(completions)} completions vs "
            f"{len(expected_calls)} expected lists"
        )
    return [function_call_reward(c, e) for c, e in zip(completions, expected_calls, strict=True)]
