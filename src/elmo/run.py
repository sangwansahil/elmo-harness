"""The closed loop: spec -> data -> baseline -> (foundry -> train -> eval ->
diagnose -> gate -> regression-promote)* -> export."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from elmo.backends import get_backend
from elmo.config import Paths
from elmo.data import load_dataset as _generic_load_dataset
from elmo.data import write_eval_jsonl, write_sft_jsonl
from elmo.eval import make_evaluator
from elmo.eval.function_calling import ScoreReport
from elmo.foundry import run_foundry
from elmo.loop import (
    GateResult,
    RegressionSuite,
    capability_vector,
    diagnose,
    evaluate_gate,
)
from elmo.roles import RoleConfig, resolve_role
from elmo.spec import TaskSpec
from elmo.storage import Storage


console = Console()


@dataclass
class IterationRecord:
    n: int
    score: ScoreReport
    regression_score: float | None
    gate: GateResult | None
    adapter_path: Path
    n_new_regression_cases: int
    foundry_accepted: int = 0
    duration_s: float = 0.0


@dataclass
class RunResult:
    run_id: str
    baseline: ScoreReport
    best: ScoreReport
    best_iteration: int
    adapter_path: Path | None
    artifact_dir: Path
    iterations: list[IterationRecord] = field(default_factory=list)


def _new_run_id() -> str:
    return f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _load_dataset(spec: TaskSpec, cache_dir: Path) -> list[dict]:
    return _generic_load_dataset(
        source=spec.dataset.source,
        max_rows=spec.dataset.max_rows,
        split=spec.dataset.split,
        cache_dir=cache_dir,
    )


def _split(
    rows: list[dict], val_frac: float = 0.1, eval_n: int = 100
) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    eval_n = min(eval_n, max(10, n // 10))
    val_n = max(8, int(n * val_frac))
    return rows[eval_n + val_n :], rows[eval_n : eval_n + val_n], rows[:eval_n]


def _merge_train_jsonl(
    synthetic: Path, raw: Path, mix_with_raw_fraction: float, out: Path
) -> Path:
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
        f"[dim]backend[/dim] {backend_name}\n"
        f"[dim]iters[/dim] {spec.budget.max_iterations}"
    )
    console.print(Panel.fit(body, title="elmo", border_style="dim"))


def _print_capability_table(records: list[IterationRecord], baseline: ScoreReport) -> None:
    t = Table(show_header=True, header_style="dim", border_style="dim", title="capabilities")
    t.add_column("capability", style="dim")
    t.add_column("baseline", justify="right")
    for r in records:
        t.add_column(f"iter {r.n}", justify="right")
    for key in ("tool_selection", "arguments", "parallel_calls", "overall"):
        row = [key.replace("_", " "), f"{getattr(baseline, key):.3f}"]
        for r in records:
            v = getattr(r.score, key)
            d = v - getattr(baseline, key)
            cell = f"{v:.3f} ({d:+.3f})"
            row.append(cell)
        t.add_row(*row)
    console.print(t)


def _promote_failures(
    final: ScoreReport,
    regression: RegressionSuite,
    eval_rows: list[dict],
    iteration: int,
    run_id: str,
) -> int:
    """Each failing eval example becomes a permanent regression case.

    Currently only function-calling eval rows have the right shape; other
    benchmarks no-op here until the suite is generalized.
    """
    if not eval_rows or "expected_calls" not in eval_rows[0]:
        return 0
    n_new = 0
    by_query = {r["query"]: r for r in eval_rows}
    for e in final.per_example:
        if e.get("c2", 0.0) >= 1.0:
            continue
        query = e.get("query", "")
        source = by_query.get(query) or by_query.get(query.split("\n")[0])
        if source is None:
            continue
        # Bucket into the most relevant capability with the same heuristic as diagnose.
        expected = e.get("expected") or []
        if len(expected) > 1:
            cap = "parallel_calls"
        elif e.get("c1", 0.0) >= 1.0:
            cap = "arguments"
        else:
            cap = "tool_selection"
        added = regression.add_failure(
            capability=cap,
            query=source["query"],
            tools=source["tools"],
            expected_calls=source["expected_calls"],
            system=source["system"],
            iteration=iteration,
            source_run_id=run_id,
        )
        if added is not None:
            n_new += 1
    return n_new


def execute(
    spec: TaskSpec,
    paths: Paths,
    progress: Callable[[str, str], None] | None = None,
    run_id: str | None = None,
) -> RunResult:
    paths.ensure()
    storage = Storage(paths.db)
    run_id = run_id or _new_run_id()
    artifact_dir = paths.runs / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(spec.backend, objective=spec.train.objective)
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

    # --- one-time setup: load data, baseline eval, regression suite -----------
    tick("data", f"loading {spec.dataset.source} (max_rows={spec.dataset.max_rows})")
    rows = _load_dataset(spec, paths.cache)
    train_rows, val_rows, eval_rows_raw = _split(rows, eval_n=spec.eval.max_examples)
    raw_train_jsonl = artifact_dir / "train.raw.jsonl"
    val_jsonl = artifact_dir / "valid.jsonl"
    eval_jsonl = artifact_dir / "eval.jsonl"
    n_train = write_sft_jsonl(train_rows, raw_train_jsonl)
    n_val = write_sft_jsonl(val_rows, val_jsonl)
    n_eval = write_eval_jsonl(eval_rows_raw, eval_jsonl)
    tick("data", f"raw: train={n_train} valid={n_val} eval={n_eval}")

    # Foundry-seed path: no public dataset, generate everything from the prompt.
    if n_eval == 0 and spec.foundry.enabled:
        planner_cfg_seed = resolve_role(
            "planner", RoleConfig(**spec.roles.planner.model_dump()) if spec.roles.planner else None
        )
        gen_cfg_seed = resolve_role(
            "generator", RoleConfig(**spec.roles.generator.model_dump()) if spec.roles.generator else None
        )
        if planner_cfg_seed and gen_cfg_seed:
            tick("foundry", "seeding from prompt (no public dataset configured)")
            seed = run_foundry(
                spec=spec,
                planner_cfg=planner_cfg_seed,
                generator_cfg=gen_cfg_seed,
                artifact_dir=artifact_dir / "seed",
                n_scenarios=spec.foundry.scenarios_per_brief,
                iteration=0,
                progress=tick,
            )
            if seed.accepted > 0:
                lines = [ln for ln in seed.sft_jsonl.read_text().splitlines() if ln.strip()]
                cut = max(8, len(lines) // 5)
                raw_train_jsonl.write_text("\n".join(lines[cut:]) + "\n")
                eval_jsonl.write_text(seed.eval_jsonl.read_text())
                eval_rows_raw = [
                    {"messages": [], "_eval": json.loads(line)}
                    for line in eval_jsonl.read_text().splitlines() if line.strip()
                ]
                n_train = sum(1 for _ in raw_train_jsonl.open())
                n_eval = sum(1 for _ in eval_jsonl.open())
                tick("data", f"foundry seed: train={n_train} eval={n_eval}")
            else:
                tick("foundry", "seed produced zero accepted rows")
        else:
            tick("foundry", "skipped seed: no planner/generator role configured")

    # Materialize the eval rows we just wrote (round-trip = same as evaluator sees)
    eval_rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line.strip()]
    evaluator = make_evaluator(spec.eval.benchmark, eval_jsonl)

    regression_path = Path.cwd() / "runs" / f"{spec.name}.regression.jsonl"
    regression = RegressionSuite(regression_path)
    if regression.cases:
        tick("regression", f"loaded {len(regression.cases)} prior cases from {regression_path.name}")

    def gen_baseline(prompts: list[str]) -> list[str]:
        return backend.generate(spec.base_model, None, prompts, max_tokens=512)

    tick("eval", "scoring baseline (untrained model)")
    baseline = evaluator.evaluate(gen_baseline, max_examples=spec.eval.max_examples)
    storage.update_run(run_id, baseline_score=baseline.overall)
    storage.record_iteration(run_id, 0, baseline.overall, None, "baseline")
    _baseline_added = _promote_failures(baseline, regression, eval_rows, 0, run_id)
    if _baseline_added:
        tick("regression", f"+{_baseline_added} new cases from baseline failures")

    # --- closed loop ----------------------------------------------------------
    target = spec.eval.target_score
    max_iters = max(1, spec.budget.max_iterations)
    iterations: list[IterationRecord] = []
    best_score = baseline
    best_iter = 0
    best_adapter: Path | None = None
    best_vector = capability_vector(baseline)
    diag_brief = ""

    planner_cfg = resolve_role(
        "planner", RoleConfig(**spec.roles.planner.model_dump()) if spec.roles.planner else None
    )
    gen_cfg = resolve_role(
        "generator", RoleConfig(**spec.roles.generator.model_dump()) if spec.roles.generator else None
    )

    for n in range(1, max_iters + 1):
        iter_t0 = time.time()
        iter_dir = artifact_dir / f"iter_{n:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        train_jsonl = raw_train_jsonl
        foundry_accepted = 0
        if spec.foundry.enabled and planner_cfg and gen_cfg:
            tick("plan", f"iter {n}: planner {planner_cfg.provider}/{planner_cfg.model}")
            result = run_foundry(
                spec=spec,
                planner_cfg=planner_cfg,
                generator_cfg=gen_cfg,
                artifact_dir=iter_dir / "foundry",
                n_scenarios=spec.foundry.scenarios_per_brief,
                iteration=n,
                baseline_notes=diag_brief,
                progress=tick,
            )
            foundry_accepted = result.accepted
            tick(
                "foundry",
                f"iter {n}: accepted {result.accepted} / rejected {result.rejected}",
            )
            train_jsonl = _merge_train_jsonl(
                synthetic=result.sft_jsonl,
                raw=raw_train_jsonl,
                mix_with_raw_fraction=spec.foundry.mix_with_raw_fraction,
                out=iter_dir / "train.merged.jsonl",
            )

        tick("train", f"iter {n}: lora fine-tune ({spec.train.max_steps or 200} steps)")
        train_result = backend.train(
            spec=spec, train_jsonl=train_jsonl, val_jsonl=val_jsonl, out_dir=iter_dir
        )
        tick("train", f"iter {n}: done — train_loss={train_result.train_loss:.4f}")

        def gen_tuned(prompts: list[str]) -> list[str]:
            return backend.generate(
                spec.base_model, train_result.adapter_path, prompts, max_tokens=512
            )

        tick("eval", f"iter {n}: scoring on held-out")
        score = evaluator.evaluate(gen_tuned, max_examples=spec.eval.max_examples)

        # Score the regression suite separately so we know about cross-iteration regressions
        regression_score: float | None = None
        if regression.cases:
            reg_path = iter_dir / "regression.eval.jsonl"
            n_reg = regression.write_eval_jsonl(reg_path)
            if n_reg > 0:
                reg_eval = FunctionCallEvaluator(reg_path)
                reg_report = reg_eval.evaluate(gen_tuned, max_examples=n_reg)
                regression_score = reg_report.overall
                tick("regression", f"iter {n}: suite score {regression_score:.3f} over {n_reg} cases")

        gate = evaluate_gate(score, best_score, epsilon=0.005)
        if gate.passed and score.overall >= best_score.overall:
            best_score, best_iter, best_adapter = score, n, train_result.adapter_path
            best_vector = capability_vector(score)
            tick("gate", f"iter {n}: passed — {gate.deltas}")
        else:
            tick("gate", f"iter {n}: {gate.reason}; keeping iter {best_iter} as best")

        # Promote failures into the regression suite, then ask the planner to diagnose
        new_cases = _promote_failures(score, regression, eval_rows, n, run_id)
        if new_cases:
            tick("regression", f"iter {n}: +{new_cases} new cases (suite={len(regression.cases)})")

        if planner_cfg and spec.foundry.enabled and n < max_iters:
            tick("diagnose", f"iter {n}: clustering failures via {planner_cfg.provider}")
            clusters = diagnose(score.per_example, planner_cfg)
            (iter_dir / "diagnose.json").write_text(
                json.dumps([{
                    "id": c.id, "capability": c.capability, "size": c.size,
                    "summary": c.summary, "corrective_brief": c.corrective_brief
                } for c in clusters], indent=2)
            )
            if clusters:
                # Hand the next iteration the largest cluster's corrective brief
                clusters.sort(key=lambda c: c.size, reverse=True)
                diag_brief = (
                    f"prior iter {n} failed on '{clusters[0].capability}' "
                    f"({clusters[0].size} cases). focus next: {clusters[0].corrective_brief}"
                )

        storage.record_iteration(
            run_id, n, score.overall, score.overall - baseline.overall, f"iter {n}"
        )
        iterations.append(IterationRecord(
            n=n, score=score, regression_score=regression_score, gate=gate,
            adapter_path=train_result.adapter_path, n_new_regression_cases=new_cases,
            foundry_accepted=foundry_accepted, duration_s=time.time() - iter_t0,
        ))
        _print_capability_table(iterations, baseline)

        if target is not None and score.overall >= target:
            tick("loop", f"iter {n}: target {target:.3f} reached. stopping.")
            break
        if len(iterations) >= 3 and all(
            abs(it.score.overall - iterations[-3].score.overall) < 0.005 for it in iterations[-3:]
        ):
            tick("loop", f"iter {n}: 3-iter plateau. stopping early.")
            break

    storage.update_run(run_id, final_score=best_score.overall, status="done")
    report = {
        "run_id": run_id,
        "spec": spec.model_dump(),
        "baseline": baseline.as_dict(),
        "best": best_score.as_dict(),
        "best_iteration": best_iter,
        "iterations": [{
            "n": it.n, "score": it.score.as_dict(), "regression_score": it.regression_score,
            "gate": {"passed": it.gate.passed if it.gate else None,
                      "deltas": it.gate.deltas if it.gate else None,
                      "regressions": it.gate.regressions if it.gate else None},
            "n_new_regression_cases": it.n_new_regression_cases,
            "foundry_accepted": it.foundry_accepted, "duration_s": round(it.duration_s, 1),
        } for it in iterations],
        "regression_suite": {"path": str(regression_path), "total_cases": len(regression.cases)},
        "best_vector": best_vector,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))

    # Append a trajectory to the local prior for future runs to retrieve from.
    try:
        from elmo.trajectory import TrajectoryStore, trajectory_from_report
        traj = trajectory_from_report(report, spec.model_dump())
        traj.planner_model = (planner_cfg.model if planner_cfg else "")
        traj.generator_model = (gen_cfg.model if gen_cfg else "")
        store = TrajectoryStore(Path.cwd() / "runs" / "trajectories.jsonl")
        store.add(traj)
        tick("trajectory", f"+1 ({traj.id}) appended to runs/trajectories.jsonl")
    except Exception as e:
        tick("trajectory", f"skipped: {e!r}")

    console.print(Panel.fit(
        f"[dim]overall[/dim]  {baseline.overall:.3f} → {best_score.overall:.3f}  "
        f"([{'green' if best_score.overall >= baseline.overall else 'red'}]"
        f"{(best_score.overall - baseline.overall):+.3f}[/])  "
        f"[dim]best iter[/dim] {best_iter}/{len(iterations)}",
        title="result", border_style="dim",
    ))
    return RunResult(
        run_id=run_id, baseline=baseline, best=best_score, best_iteration=best_iter,
        adapter_path=best_adapter, artifact_dir=artifact_dir, iterations=iterations,
    )
