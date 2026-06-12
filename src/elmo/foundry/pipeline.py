"""End-to-end foundry pipeline — plan → generate → filter → log."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from elmo.data.xlam import _row_to_messages  # internal but stable
from elmo.foundry.filter import filter_row
from elmo.foundry.generator import generate_row
from elmo.foundry.planner import DataBrief, build_brief
from elmo.foundry.provenance import ProvenanceLog
from elmo.roles import RoleConfig
from elmo.spec import TaskSpec


@dataclass
class FoundryResult:
    brief: DataBrief
    accepted: int
    rejected: int
    generator_failed: int
    sft_jsonl: Path
    eval_jsonl: Path
    provenance_jsonl: Path
    brief_json: Path


def _write_sft(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
            n += 1
    return n


def _write_eval(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r["_eval"]) + "\n")
            n += 1
    return n


def run_foundry(
    spec: TaskSpec,
    planner_cfg: RoleConfig,
    generator_cfg: RoleConfig,
    artifact_dir: Path,
    n_scenarios: int,
    iteration: int = 0,
    baseline_notes: str = "",
    progress: Callable[[str, str], None] | None = None,
) -> FoundryResult:
    """Plan a brief, generate rows from each scenario, filter, and persist."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prov = ProvenanceLog(artifact_dir / "provenance.jsonl")

    def tick(msg: str) -> None:
        if progress:
            progress("foundry", msg)

    tick(f"planning {n_scenarios} scenarios with {planner_cfg.provider}/{planner_cfg.model}")
    brief = build_brief(
        spec=spec,
        planner_cfg=planner_cfg,
        n_scenarios=n_scenarios,
        iteration=iteration,
        baseline_notes=baseline_notes,
    )
    (artifact_dir / "brief.json").write_text(brief.model_dump_json(indent=2))
    tick(f"brief {brief.id}: {len(brief.scenarios)} scenarios")

    accepted: list[dict] = []
    rejected = 0
    gen_failed = 0

    for scenario in brief.scenarios:
        try:
            xlam_row, telemetry = generate_row(scenario, generator_cfg)
        except Exception as e:
            gen_failed += 1
            tick(f"gen failed for {scenario.id}: {e!r}")
            continue

        report = filter_row(xlam_row, allow_refusal=True)
        row_id = f"r_{uuid.uuid4().hex[:10]}"

        if not report.passed:
            rejected += 1
            prov.write(
                row_id=row_id,
                scenario_id=scenario.id,
                brief_id=brief.id,
                planner_model=brief.planner_model,
                generator_model=telemetry["generator_model"],
                generator_tokens=(telemetry["prompt_tokens"], telemetry["completion_tokens"]),
                verifier_passed=False,
                verifier_reasons=report.reasons,
                seed_prompt_hash=ProvenanceLog.hash_prompt(scenario.scenario),
            )
            continue

        # Reuse the xLAM formatter to produce chat-format messages + _eval block
        # _row_to_messages expects JSON-stringified tools/answers, so we re-stringify.
        encoded = {
            "tools": json.dumps(xlam_row["tools"]),
            "answers": json.dumps(xlam_row["answers"]),
            "query": xlam_row["query"],
        }
        formatted = _row_to_messages(encoded)
        if formatted is None:
            rejected += 1
            prov.write(
                row_id=row_id,
                scenario_id=scenario.id,
                brief_id=brief.id,
                planner_model=brief.planner_model,
                generator_model=telemetry["generator_model"],
                generator_tokens=(telemetry["prompt_tokens"], telemetry["completion_tokens"]),
                verifier_passed=False,
                verifier_reasons=["xlam-format conversion failed"],
                seed_prompt_hash=ProvenanceLog.hash_prompt(scenario.scenario),
            )
            continue

        accepted.append(formatted)
        prov.write(
            row_id=row_id,
            scenario_id=scenario.id,
            brief_id=brief.id,
            planner_model=brief.planner_model,
            generator_model=telemetry["generator_model"],
            generator_tokens=(telemetry["prompt_tokens"], telemetry["completion_tokens"]),
            verifier_passed=True,
            verifier_reasons=[],
            seed_prompt_hash=ProvenanceLog.hash_prompt(scenario.scenario),
            extra={"capability": scenario.capability, "domain": scenario.domain},
        )

    sft_path = artifact_dir / "foundry_train.jsonl"
    eval_path = artifact_dir / "foundry_eval.jsonl"
    _write_sft(accepted, sft_path)
    _write_eval(accepted, eval_path)

    tick(f"accepted {len(accepted)} · rejected {rejected} · gen_failed {gen_failed}")

    return FoundryResult(
        brief=brief,
        accepted=len(accepted),
        rejected=rejected,
        generator_failed=gen_failed,
        sft_jsonl=sft_path,
        eval_jsonl=eval_path,
        provenance_jsonl=artifact_dir / "provenance.jsonl",
        brief_json=artifact_dir / "brief.json",
    )
