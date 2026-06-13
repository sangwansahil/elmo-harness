"""Diagnose step — cluster eval failures by capability + theme.

Two-pass approach:
  1. Bucket failures by primary capability (cheap, deterministic).
  2. For each non-empty bucket, ask the planner to write a short failure-mode
     summary and propose a corrective data brief (handed to the next iteration).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from elmo.providers import CompletionRequest, get_provider
from elmo.roles import RoleConfig


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class FailureCluster:
    id: str
    capability: str
    size: int
    summary: str
    corrective_brief: str
    examples: list[dict] = field(default_factory=list)


SYSTEM_PROMPT = (
    "You are the diagnose agent. Given a list of failures from a fine-tuned "
    "model, write a brief failure-mode summary and propose how to fix it with "
    "targeted training data. Output JSON only — no markdown fences, no prose."
)


def _user_prompt(capability: str, examples: list[dict]) -> str:
    pruned = []
    for e in examples[:12]:
        pruned.append({
            "query": e.get("query", "")[:160],
            "expected": e.get("expected", []),
            "predicted": e.get("predicted", []),
            "c1": e.get("c1"),
            "c2": e.get("c2"),
        })
    return (
        f"Capability with failures: {capability}\n\n"
        f"Sample failures (up to 12):\n"
        f"{json.dumps(pruned, indent=2)}\n\n"
        f"Output JSON with this shape:\n"
        f"{{\n"
        f'  "summary": "<one short paragraph describing the failure mode>",\n'
        f'  "corrective_brief": "<one paragraph briefing the data generator on what to make next>"\n'
        f"}}\n"
    )


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return {}
    try:
        obj = json.loads(text[start:end])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def diagnose(
    per_example: list[dict],
    planner_cfg: RoleConfig,
    capabilities: list[str] | None = None,
    threshold: float = 1.0,
) -> list[FailureCluster]:
    """Return one FailureCluster per capability that has failures.

    `per_example` is the ScoreReport.per_example list. A failure is c2 < threshold.
    """
    by_cap: dict[str, list[dict]] = {}
    for e in per_example:
        score = float(e.get("c2", 0.0))
        if score < threshold:
            cap = _infer_capability(e, capabilities)
            by_cap.setdefault(cap, []).append(e)

    clusters: list[FailureCluster] = []
    if not by_cap:
        return clusters

    provider = get_provider(planner_cfg.provider)
    for cap, examples in by_cap.items():
        req = CompletionRequest(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(cap, examples)},
            ],
            model=planner_cfg.model,
            temperature=0.3,
            max_tokens=1024,
        )
        try:
            resp = provider.complete(req)
            obj = _extract_json_object(resp.text)
        except Exception as e:
            obj = {"summary": f"diagnose call failed: {e!r}", "corrective_brief": ""}
        clusters.append(
            FailureCluster(
                id=f"fc_{uuid.uuid4().hex[:8]}",
                capability=cap,
                size=len(examples),
                summary=obj.get("summary", "")[:600],
                corrective_brief=obj.get("corrective_brief", "")[:1000],
                examples=examples,
            )
        )
    return clusters


def _infer_capability(example: dict, capabilities: list[str] | None) -> str:
    """Bucket an example into one capability — Phase 2 heuristic.

    For function calling: parallel-calls failures map to 'parallel_calls' if the
    expected has >1 call; otherwise 'arguments' if the name matched (c1==1) but
    args didn't; otherwise 'tool_selection'.
    """
    expected = example.get("expected") or []
    if len(expected) > 1:
        return "parallel_calls"
    c1 = float(example.get("c1", 0.0))
    if c1 >= 1.0:
        return "arguments"
    return "tool_selection"
