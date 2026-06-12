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
    backend = detect_backend()
    console.print(f"[dim]backend[/dim]   {backend}")
    for mod in ("mlx_lm", "datasets", "huggingface_hub", "pydantic", "typer", "yaml"):
        try:
            __import__(mod)
            console.print(f"[green]ok[/green]        {mod}")
        except ImportError as e:
            console.print(f"[red]missing[/red]   {mod}  ({e})")


if __name__ == "__main__":
    app()
