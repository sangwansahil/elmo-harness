"""elmo CLI — typer entry points."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from elmo import __version__
from elmo.config import Paths, detect_backend
from elmo.spec import load_spec
from elmo.storage import Storage


app = typer.Typer(
    help="prompt → expert slm. fine-tune small llms locally with a closed loop.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the elmo version."""
    console.print(f"elmo {__version__}")


@app.command()
def init() -> None:
    """Set up `.elmo/` and copy the example task spec into the cwd."""
    paths = Paths.from_cwd()
    paths.ensure()
    example_src = Path(__file__).parent.parent.parent / "examples" / "function-calling.yaml"
    example_dst = Path.cwd() / "examples" / "function-calling.yaml"
    if example_src.exists() and not example_dst.exists():
        example_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(example_src, example_dst)

    backend = detect_backend()
    console.print(f"[dim]root[/dim]      {paths.root}")
    console.print(f"[dim]db[/dim]        {paths.db}")
    console.print(f"[dim]runs[/dim]      {paths.runs}")
    console.print(f"[dim]backend[/dim]   {backend}")
    if backend == "none":
        console.print(
            "[yellow]no local training backend detected. "
            "elmo needs apple silicon (mlx) or nvidia (unsloth).[/yellow]"
        )
    console.print("\nnext: [bold]elmo run examples/function-calling.yaml[/bold]")


@app.command()
def run(
    spec_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="parse spec and exit without training."),
) -> None:
    """Run the loop end-to-end for a task spec."""
    spec = load_spec(spec_path)
    if dry_run:
        console.print(spec.model_dump_json(indent=2))
        return
    # Heavy import deferred so `elmo --help` is fast even without mlx installed.
    from elmo.run import execute

    paths = Paths.from_cwd()
    execute(spec, paths)


@app.command(name="runs")
def list_runs(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """List recent runs."""
    paths = Paths.from_cwd()
    paths.ensure()
    storage = Storage(paths.db)
    rows = storage.list_runs(limit=limit)
    if not rows:
        console.print("[dim]no runs yet. try:[/dim] elmo run examples/function-calling.yaml")
        return
    t = Table(show_header=True, header_style="dim", border_style="dim")
    for col in ("id", "task", "model", "status", "base", "final", "Δ", "when"):
        t.add_column(col, style="dim" if col in {"id", "when"} else None)
    for r in rows:
        b, f = r.get("baseline_score"), r.get("final_score")
        delta = (f - b) if (b is not None and f is not None) else None
        when = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        t.add_row(
            r["id"][:24],
            r["task_name"],
            r["base_model"].split("/")[-1][:24],
            r["status"],
            f"{b:.3f}" if b is not None else "—",
            f"{f:.3f}" if f is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—",
            when,
        )
    console.print(t)


@app.command()
def doctor() -> None:
    """Diagnose the local environment."""
    from elmo.providers import KNOWN_PROVIDERS, list_available
    from elmo.roles import resolve_role

    backend = detect_backend()
    console.print(f"[dim]backend[/dim]   {backend}")
    for mod in ("mlx_lm", "datasets", "huggingface_hub", "pydantic", "typer", "yaml"):
        try:
            __import__(mod)
            console.print(f"[green]ok[/green]        {mod}")
        except ImportError as e:
            console.print(f"[red]missing[/red]   {mod}  ({e})")

    console.print()
    available = set(list_available())
    for name, cfg in KNOWN_PROVIDERS.items():
        mark = "[green]on[/green]" if name in available else "[dim]off[/dim]"
        key = cfg.env_key or "(no key needed)"
        console.print(f"{mark:>16}  {name:<12}  {key}")

    console.print()
    for role in ("planner", "generator", "judge"):
        r = resolve_role(role)  # type: ignore[arg-type]
        if r:
            console.print(f"[dim]{role:>9s}[/dim]   {r.provider}/{r.model}")
        else:
            console.print(f"[dim]{role:>9s}[/dim]   [yellow]not configured[/yellow]")


@app.command()
def plan(
    spec_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    n: int = typer.Option(5, "-n", help="number of scenarios to plan"),
) -> None:
    """Dry-run the planner: produce a brief and print it without generating data."""
    from elmo.foundry.planner import build_brief
    from elmo.roles import resolve_role
    from elmo.spec import load_spec

    spec = load_spec(spec_path)
    planner_cfg = resolve_role("planner")
    if planner_cfg is None:
        console.print("[red]no planner role configured.[/red] try: elmo doctor")
        raise typer.Exit(2)
    console.print(f"[dim]planner[/dim] {planner_cfg.provider}/{planner_cfg.model}")
    brief = build_brief(spec=spec, planner_cfg=planner_cfg, n_scenarios=n)
    console.print(brief.model_dump_json(indent=2))


@app.command()
def regression(
    action: str = typer.Argument("list", help="list | show | clear"),
    task: str = typer.Option("function-calling", "--task", "-t"),
) -> None:
    """Inspect or clear the regression suite for a task."""
    from elmo.loop.regression import RegressionSuite

    path = Path.cwd() / "runs" / f"{task}.regression.jsonl"
    suite = RegressionSuite(path)
    if action == "list":
        by_cap = suite.by_capability()
        if not by_cap:
            console.print("[dim]no cases yet.[/dim]")
            return
        t = Table(show_header=True, header_style="dim", border_style="dim")
        t.add_column("capability")
        t.add_column("cases", justify="right")
        t.add_column("fixed", justify="right")
        for cap, cases in by_cap.items():
            fixed = sum(1 for c in cases if c.fixed_in_iter is not None)
            t.add_row(cap, str(len(cases)), str(fixed))
        console.print(t)
    elif action == "show":
        for c in suite.cases[:50]:
            console.print(
                f"[dim]{c.id}[/dim] [dim]{c.capability}[/dim] iter={c.first_seen_iter} "
                f"fixed={c.fixed_in_iter}\n  {c.query[:160]}"
            )
    elif action == "clear":
        path.unlink(missing_ok=True)
        console.print(f"[yellow]cleared[/yellow] {path}")
    else:
        console.print(f"[red]unknown action: {action}[/red] (use list, show, or clear)")
        raise typer.Exit(2)


@app.command()
def providers() -> None:
    """List supported providers and which are currently configured."""
    from elmo.providers import KNOWN_PROVIDERS, list_available

    available = set(list_available())
    t = Table(show_header=True, header_style="dim", border_style="dim")
    t.add_column("provider")
    t.add_column("base url", style="dim")
    t.add_column("env var", style="dim")
    t.add_column("status")
    for name, cfg in KNOWN_PROVIDERS.items():
        status = "[green]on[/green]" if name in available else "[dim]off[/dim]"
        t.add_row(name, cfg.base_url, cfg.env_key or "—", status)
    console.print(t)


if __name__ == "__main__":
    app()
