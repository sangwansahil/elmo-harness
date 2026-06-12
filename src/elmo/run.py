"""The Phase 0 loop: spec -> data -> baseline eval -> train -> eval -> record."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from elmo.backends import get_backend
from elmo.config import Paths
from elmo.data.xlam import (
    load_xlam_function_calling,
    write_eval_jsonl,
    write_sft_jsonl,
)
from elmo.eval.function_calling import FunctionCallEvaluator, ScoreReport
from elmo.foundry import run_foundry
from elmo.roles import RoleConfig, resolve_role
from elmo.spec import TaskSpec
from elmo.storage import Storage


console = Console()


@dataclass
class RunResult:
    run_id: str
    baseline: ScoreReport
    final: ScoreReport
    adapter_path: Path | None
    artifact_dir: Path


def _new_run_id() -> str:
    return f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _load_dataset(spec: TaskSpec, cache_dir: Path) -> list[dict]:
    if spec.dataset.source == "hf:Salesforce/xlam-function-calling-60k":
        return load_xlam_function_calling(
            max_rows=spec.dataset.max_rows,
            split=spec.dataset.split,
            cache_dir=cache_dir,
        )
    raise NotImplementedError(
        f"phase 0 only ships the xlam loader. got: {spec.dataset.source}"
    )


def _split(rows: list[dict], val_frac: float = 0.1, eval_n: int = 100) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    eval_n = min(eval_n, max(10, n // 10))
    val_n = max(8, int(n * val_frac))
    eval_rows = rows[:eval_n]
    val_rows = rows[eval_n : eval_n + val_n]
    train_rows = rows[eval_n + val_n :]
    return train_rows, val_rows, eval_rows


def _merge_train_jsonl(
    synthetic: Path, raw: Path, mix_with_raw_fraction: float, out: Path
) -> Path:
    """Concatenate synthetic + raw rows by ratio. Synthetic-first ordering."""
    syn_lines = [ln for ln in synthetic.read_text().splitlines() if ln.strip()]
    raw_lines = [ln for ln in raw.read_text().splitlines() if ln.strip()]
    target_raw = int(len(syn_lines) * mix_with_raw_fraction / max(1e-6, 1 - mix_with_raw_fraction))
    merged = syn_lines + raw_lines[:target_raw]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(merged) + "\n")
    return out


def _print_header(spec: TaskSpec, run_id: str, backend_name: str) -> None:
    body = (
        f"[dim]run[/dim] {run_id}\n"
        f"[dim]task[/dim] {spec.name}\n"
        f"[dim]base[/dim] {spec.base_model}\n"
        f"[dim]backend[/dim] {backend_name}"
    )
    console.print(Panel.fit(body, title="elmo", border_style="dim"))


def _print_scores(label: str, baseline: ScoreReport, final: ScoreReport | None = None) -> None:
    t = Table(title=label, show_header=True, header_style="dim", border_style="dim")
    t.add_column("capability", style="dim")
    t.add_column("baseline", justify="right")
    if final is not None:
        t.add_column("final", justify="right")
        t.add_column("Δ", justify="right")
    for key, attr in [
        ("tool selection", "tool_selection"),
        ("arguments", "arguments"),
        ("parallel calls", "parallel_calls"),
        ("overall", "overall"),
    ]:
        b = getattr(baseline, attr)
        row = [key, f"{b:.3f}"]
        if final is not None:
            f = getattr(final, attr)
            d = f - b
            sign = "+" if d >= 0 else ""
            row += [f"{f:.3f}", f"{sign}{d:.3f}"]
        t.add_row(*row)
    console.print(t)


def execute(
    spec: TaskSpec,
    paths: Paths,
    progress: Callable[[str, str], None] | None = None,
) -> RunResult:
    paths.ensure()
    storage = Storage(paths.db)
    run_id = _new_run_id()
    artifact_dir = paths.runs / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(spec.backend)
    storage.create_run(
        run_id=run_id,
        task_name=spec.name,
        spec_json=spec.model_dump_json(),
        base_model=spec.base_model,
        backend=backend.name,
        artifact_dir=str(artifact_dir),
    )
    _print_header(spec, run_id, backend.name)

    def tick(stage: str, msg: str) -> None:
        storage.log_event(run_id, stage, msg)
        console.print(f"[dim]{stage:8s}[/dim] {msg}")
        if progress:
            progress(stage, msg)

    tick("data", f"loading {spec.dataset.source} (max_rows={spec.dataset.max_rows})")
    rows = _load_dataset(spec, paths.cache)
    train_rows, val_rows, eval_rows = _split(rows, eval_n=spec.eval.max_examples)
    train_jsonl = artifact_dir / "train.jsonl"
    val_jsonl = artifact_dir / "valid.jsonl"
    eval_jsonl = artifact_dir / "eval.jsonl"
    n_train = write_sft_jsonl(train_rows, train_jsonl)
    n_val = write_sft_jsonl(val_rows, val_jsonl)
    n_eval = write_eval_jsonl(eval_rows, eval_jsonl)
    tick("data", f"raw xlam: train={n_train} valid={n_val} eval={n_eval}")

    if spec.foundry.enabled:
        planner_cfg = resolve_role(
            "planner",
            RoleConfig(**spec.roles.planner.model_dump()) if spec.roles.planner else None,
        )
        gen_cfg = resolve_role(
            "generator",
            RoleConfig(**spec.roles.generator.model_dump()) if spec.roles.generator else None,
        )
        if planner_cfg is None or gen_cfg is None:
            tick("foundry", "skipped — no provider keys for planner or generator")
        else:
            tick("plan", f"planner {planner_cfg.provider}/{planner_cfg.model}")
            foundry_dir = artifact_dir / "foundry"
            result = run_foundry(
                spec=spec,
                planner_cfg=planner_cfg,
                generator_cfg=gen_cfg,
                artifact_dir=foundry_dir,
                n_scenarios=spec.foundry.scenarios_per_brief,
                iteration=0,
                progress=tick,
            )
            tick(
                "foundry",
                f"accepted {result.accepted} / rejected {result.rejected} / "
                f"gen_failed {result.generator_failed}",
            )
            train_jsonl = _merge_train_jsonl(
                synthetic=result.sft_jsonl,
                raw=train_jsonl,
                mix_with_raw_fraction=spec.foundry.mix_with_raw_fraction,
                out=artifact_dir / "train.merged.jsonl",
            )
            tick("data", f"training on {sum(1 for _ in train_jsonl.open())} merged rows")

    evaluator = FunctionCallEvaluator(eval_jsonl)

    def gen_baseline(prompts: list[str]) -> list[str]:
        return backend.generate(spec.base_model, None, prompts, max_tokens=512)

    tick("eval", "scoring baseline (untrained model)")
    baseline = evaluator.evaluate(gen_baseline, max_examples=spec.eval.max_examples)
    storage.update_run(run_id, baseline_score=baseline.overall)
    storage.record_iteration(run_id, 0, baseline.overall, None, "baseline")
    _print_scores("baseline", baseline)

    tick("train", f"lora fine-tune ({spec.train.max_steps or 200} steps)")
    train_result = backend.train(
        spec=spec,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        out_dir=artifact_dir,
    )
    tick("train", f"done — train_loss={train_result.train_loss:.4f}")

    def gen_tuned(prompts: list[str]) -> list[str]:
        return backend.generate(
            spec.base_model, train_result.adapter_path, prompts, max_tokens=512
        )

    tick("eval", "scoring fine-tuned model")
    final = evaluator.evaluate(gen_tuned, max_examples=spec.eval.max_examples)
    storage.update_run(run_id, final_score=final.overall, status="done")
    storage.record_iteration(
        run_id, 1, final.overall, final.overall - baseline.overall, "post-sft"
    )
    _print_scores("result", baseline, final)

    (artifact_dir / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "spec": spec.model_dump(),
                "baseline": baseline.as_dict(),
                "final": final.as_dict(),
                "delta": round(final.overall - baseline.overall, 4),
                "train_loss": train_result.train_loss,
                "adapter": str(train_result.adapter_path),
            },
            indent=2,
        )
    )
    console.print(
        Panel.fit(
            f"[dim]overall[/dim]  {baseline.overall:.3f} → {final.overall:.3f}  "
            f"([{'green' if final.overall >= baseline.overall else 'red'}]"
            f"{(final.overall - baseline.overall):+.3f}[/])",
            title="result",
            border_style="dim",
        )
    )
    return RunResult(
        run_id=run_id,
        baseline=baseline,
        final=final,
        adapter_path=train_result.adapter_path,
        artifact_dir=artifact_dir,
    )
