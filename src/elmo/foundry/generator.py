"""Generator — turn one Scenario into one xLAM-format training row.

The generator is the cheap model. We send each scenario individually so failures
are row-level and parallelizable. The output is the same shape as our
xLAM-format rows (tools, query, answers), so the rest of the pipeline is reused.
"""

from __future__ import annotations

import json
import re

from elmo.providers import CompletionRequest, get_provider
from elmo.roles import RoleConfig
from elmo.foundry.planner import Scenario


SYSTEM_PROMPT = (
    "You are the generator. You convert one scenario into one realistic "
    "function-calling training row. Output JSON only — no markdown fences, no prose."
)


def _user_prompt(scenario: Scenario) -> str:
    return (
        f"Scenario:\n{scenario.model_dump_json(indent=2)}\n\n"
        f"Produce ONE JSON object with this exact shape:\n"
        f"{{\n"
        f'  "tools": [/* the tool_schema(s), as a JSON array of tool defs */],\n'
        f'  "query": "/* a natural user message that requires the expected calls */",\n'
        f'  "answers": [/* the expected calls */]\n'
        f"}}\n\n"
        f"Guidance:\n"
        f"- 'query' must read like a real user message, not a paraphrase of the scenario.\n"
        f"- 'tools' may include 1-3 plausible siblings to test selection ability.\n"
        f"- 'answers' must match the expected_calls in name and arguments.\n"
        f"- If the scenario tests refusal or ambiguity, 'answers' may be an empty array.\n"
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in generator output")
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
        raise ValueError("unbalanced JSON object in generator output")
    obj = json.loads(text[start:end])
    if not isinstance(obj, dict):
        raise ValueError("generator output is not a JSON object")
    return obj


def generate_row(scenario: Scenario, gen_cfg: RoleConfig) -> tuple[dict, dict]:
    """Run the generator once for a scenario. Returns (xlam_row, telemetry)."""
    provider = get_provider(gen_cfg.provider)
    req = CompletionRequest(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(scenario)},
        ],
        model=gen_cfg.model,
        temperature=0.4,
        max_tokens=2048,
    )
    resp = provider.complete(req)
    row = _extract_json_object(resp.text)
    telemetry = {
        "generator_model": resp.model,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
    }
    return row, telemetry
