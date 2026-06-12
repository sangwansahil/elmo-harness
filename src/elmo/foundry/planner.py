"""Planner — produces a DataBrief: a list of scenarios to generate, weighted by
capability priorities.

The strong-model planner outputs JSON only. We tolerate JSON wrapped in
``` fences or in prose, and parse it back into typed Scenario objects.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from elmo.providers import CompletionRequest, get_provider
from elmo.roles import RoleConfig
from elmo.spec import TaskSpec


class Scenario(BaseModel):
    id: str
    capability: str
    rationale: str = ""
    domain: str = ""
    tool_schema: dict | list = Field(default_factory=dict)
    scenario: str = ""
    expected_calls: list[dict] = Field(default_factory=list)


class DataBrief(BaseModel):
    id: str
    iteration: int
    task_name: str
    scenarios: list[Scenario]
    planner_model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: str = ""


SYSTEM_PROMPT = (
    "You are the planner for elmo, an open-source fine-tuning harness. "
    "You design diverse, edge-case-rich scenarios for synthetic training data. "
    "Output JSON only — no markdown fences, no prose."
)


def _user_prompt(spec: TaskSpec, n: int, baseline_notes: str = "") -> str:
    caps = "\n".join(
        f"- {c.name} (weight {c.weight}, verifier {c.verifier}): {c.description}"
        for c in spec.capabilities
    )
    return (
        f"Task: {spec.name}\n"
        f"Original prompt: {spec.prompt}\n"
        f"Base model: {spec.base_model}\n\n"
        f"Capabilities to train:\n{caps}\n\n"
        f"Iteration context: {baseline_notes or 'first iteration, no prior eval'}\n\n"
        f"Produce a JSON array of exactly {n} scenarios. Distribute scenarios across "
        f"capabilities according to their weights. Vary domain (weather, finance, code, "
        f"travel, biology, etc.), difficulty, and tool-catalog size. Include some "
        f"adversarial cases: ambiguous queries, overlapping tool names, missing required "
        f"arguments, parallel-call opportunities.\n\n"
        f"Output JSON array only:\n"
        f"[{{\n"
        f'  "id": "s_001",\n'
        f'  "capability": "<one of the capability names>",\n'
        f'  "rationale": "<one short sentence>",\n'
        f'  "domain": "<short label>",\n'
        f'  "tool_schema": [{{"name": "...", "description": "...", "parameters": {{...}}}}],\n'
        f'  "scenario": "<short description of the user situation>",\n'
        f'  "expected_calls": [{{"name": "...", "arguments": {{...}}}}]\n'
        f"}}]\n"
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    # Strip a leading code fence if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Find the first '[' and the matching ']'
    start = text.find("[")
    if start == -1:
        raise ValueError("no JSON array found in planner output")
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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("unbalanced JSON array in planner output")
    arr = json.loads(text[start:end])
    if not isinstance(arr, list):
        raise ValueError("planner output is not a JSON array")
    return arr


def build_brief(
    spec: TaskSpec,
    planner_cfg: RoleConfig,
    n_scenarios: int,
    iteration: int = 0,
    baseline_notes: str = "",
) -> DataBrief:
    provider = get_provider(planner_cfg.provider)
    req = CompletionRequest(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(spec, n_scenarios, baseline_notes)},
        ],
        model=planner_cfg.model,
        temperature=0.7,
        max_tokens=8192,
    )
    resp = provider.complete(req)
    raw = _extract_json_array(resp.text)

    scenarios: list[Scenario] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        item.setdefault("id", f"s_{i + 1:03d}")
        try:
            scenarios.append(Scenario.model_validate(item))
        except Exception:
            continue
    return DataBrief(
        id=f"brief_{uuid.uuid4().hex[:10]}",
        iteration=iteration,
        task_name=spec.name,
        scenarios=scenarios,
        planner_model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
