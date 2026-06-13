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
def trajectory(
    action: str = typer.Argument("list", help="list | publish | fetch | search"),
    repo: str = typer.Option("sangwansahil/elmo-trajectories", "--repo"),
    spec_path: Path | None = typer.Option(None, "--spec", help="for `search`"),
) -> None:
    """Inspect / sync the local trajectory prior."""
    from elmo.trajectory import TrajectoryStore, fetch_from_hf, publish_to_hf

    path = Path.cwd() / "runs" / "trajectories.jsonl"
    store = TrajectoryStore(path)
    if action == "list":
        trajs = store.all()
        if not trajs:
            console.print("[dim]no trajectories yet — finish a run first.[/dim]")
            return
        t = Table(show_header=True, header_style="dim", border_style="dim")
        for col in ("id", "task", "base", "obj", "iters", "Δ", "regs"):
            t.add_column(col, style="dim" if col == "id" else None)
        for tr in trajs:
            t.add_row(
                tr.id, tr.task_name, tr.base_model.split("/")[-1][:18],
                tr.objective, str(tr.n_iterations),
                f"{tr.delta:+.3f}", str(tr.regression_suite_size),
            )
        console.print(t)
    elif action == "publish":
        try:
            url = publish_to_hf(store.all(), repo_id=repo)
            console.print(f"[green]published[/green] {len(store.all())} trajectories → {url}")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)
    elif action == "fetch":
        try:
            fetched = fetch_from_hf(repo_id=repo)
            for tr in fetched:
                store.add(tr)
            console.print(f"[green]fetched[/green] {len(fetched)} trajectories from {repo}")
        except (RuntimeError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)
    elif action == "search":
        if not spec_path:
            console.print("[red]search needs --spec <path>[/red]")
            raise typer.Exit(2)
        from elmo.spec import load_spec
        from elmo.trajectory import trajectory_from_report
        spec = load_spec(spec_path)
        query = trajectory_from_report({"baseline": {}, "best": {}, "iterations": []}, spec.model_dump())
        hits = store.search(query, k=5)
        if not hits:
            console.print("[dim]no relevant trajectories.[/dim]")
            return
        for score, tr in hits:
            console.print(f"[dim]{score:.2f}[/dim]  {tr.task_name} · {tr.base_model.split('/')[-1]} · Δ {tr.delta:+.3f}")
    else:
        console.print(f"[red]unknown action: {action}[/red]")
        raise typer.Exit(2)


@app.command()
def cache(action: str = typer.Argument("stats", help="stats | clear")) -> None:
    """Inspect or clear the local completion cache."""
    from elmo.cache import CompletionCache

    db_path = Path.cwd() / ".elmo" / "cache" / "completions.db"
    cache = CompletionCache(db_path)
    if action == "stats":
        s = cache.stats()
        console.print(f"[dim]entries[/dim]                    {s['entries']}")
        console.print(f"[dim]total cache hits[/dim]           {s['total_hits']}")
        console.print(f"[dim]prompt tokens saved[/dim]        {s['prompt_tokens_saved']:,}")
        console.print(f"[dim]completion tokens saved[/dim]    {s['completion_tokens_saved']:,}")
        console.print(f"[dim]db[/dim]                         {db_path}")
    elif action == "clear":
        n = cache.clear()
        console.print(f"[yellow]cleared[/yellow] {n} entries")
    else:
        console.print(f"[red]unknown action: {action}[/red] (use stats or clear)")
        raise typer.Exit(2)


@app.command()
def preset(
    action: str = typer.Argument("list", help="list | apply"),
    name: str | None = typer.Argument(None, help="preset name (when action=apply)"),
) -> None:
    """List or apply a named role-config preset (e.g. free-openrouter)."""
    from elmo.presets import PRESETS, list_presets

    if action == "list":
        for n in list_presets():
            console.print(f"[dim]preset[/dim]  {n}")
            for role, cfg in PRESETS[n].items():
                console.print(f"  [dim]{role:>9s}[/dim]  {cfg.provider}/{cfg.model}")
        return
    if action == "apply":
        if not name:
            console.print("[red]preset apply requires a name[/red]")
            raise typer.Exit(2)
        if name not in PRESETS:
            console.print(f"[red]unknown preset: {name}[/red]. choose from: {list_presets()}")
            raise typer.Exit(2)
        import yaml

        out = Path.cwd() / ".elmo" / "roles.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(
            {role: cfg.model_dump() for role, cfg in PRESETS[name].items()},
            sort_keys=False,
        ))
        console.print(f"[green]applied[/green] preset '{name}' → {out}")
        console.print("export the keys it expects, then re-run `elmo doctor` to confirm.")
    else:
        console.print(f"[red]unknown action: {action}[/red] (use list or apply)")
        raise typer.Exit(2)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7777, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the elmo daemon — REST API + WebSocket + local web UI."""
    try:
        from elmo.server import serve as _serve
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    console.print(f"[dim]elmo daemon[/dim] http://{host}:{port}")
    _serve(host=host, port=port, reload=reload)


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
