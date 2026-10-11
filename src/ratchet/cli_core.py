"""Track A commands: init, eval, evolve, history, diff, rollback, memory."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ratchet.config import ConfigError, get_settings

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def info(ping: bool = typer.Option(False, "--ping", help="Also check DB connectivity.")) -> None:
    """Show the active namespace, databases and model routing (no secrets)."""
    try:
        s = get_settings()
    except ConfigError as e:
        console.print(f"[red]config error:[/red] {e}")
        raise typer.Exit(1) from e
    t = Table(title="ratchet", show_header=False)
    t.add_row("namespace", s.ns)
    t.add_row("state db", s.db_name)
    t.add_row("scratch db", s.scratch_db_name)
    t.add_row("planner", s.planner_model)
    t.add_row("critic", s.critic_model)
    t.add_row("proposer", s.proposer_model)
    for label, value in (
        ("MONGODB_URI_RW", s.mongodb_uri_rw),
        ("MONGODB_URI_RO", s.mongodb_uri_ro),
        ("ANTHROPIC_API_KEY", s.anthropic_api_key),
        ("OPENAI_API_KEY", s.openai_api_key),
        ("GOOGLE_API_KEY", s.google_api_key),
        ("VOYAGE_API_KEY", s.voyage_api_key),
    ):
        t.add_row(label, "[green]set[/green]" if value else "[dim]unset[/dim]")
    console.print(t)
    if ping:
        from ratchet import db

        for label, ok in db.ping().items():
            console.print(f"ping {label}: " + ("[green]ok[/green]" if ok else "[red]failed[/red]"))


def _fail(msg: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {msg}")
    return typer.Exit(1)


@app.command()
def init(
    no_vector: bool = typer.Option(False, "--no-vector", help="Skip Atlas Vector Search indexes."),
) -> None:
    """Create collections, indexes (incl. vector) and the gen-0 harness in `ratchet_<ns>`."""
    from ratchet import db
    from ratchet.harness.init_db import init_database

    try:
        s = get_settings()
        rep = init_database(db.get_db(), s, vector=not no_vector)
    except ConfigError as e:
        raise _fail(str(e)) from e
    console.print(f"database [bold]{rep.database}[/bold]")
    console.print(
        f"  collections created: {', '.join(rep.created_collections) or 'none (all exist)'}"
    )
    console.print(f"  indexes ensured: {len(rep.indexes)}")
    for name, status in rep.vector_indexes.items():
        colour = "yellow" if status.startswith("failed") else "green"
        console.print(f"  vector index on {name}: [{colour}]{status}[/{colour}]")
    verb = "created" if rep.head_created else "already at"
    console.print(f"  harness head {verb} [bold]{rep.head}[/bold]")


@app.command()
def pin(
    reason: str = typer.Option(..., "--reason", help="Why the scorer/gate source changed."),
) -> None:
    """Human-only: re-pin the head to the current scorer and gate source hashes."""
    from ratchet import db
    from ratchet.gate.hashes import gate_ref, scorer_ref
    from ratchet.harness.store import HarnessStore, StoreError

    try:
        doc = HarnessStore(db.get_db()).repin(
            scorer_ref().model_dump(), gate_ref().model_dump(), f"re-pin: {reason}"
        )
    except (ConfigError, StoreError) as e:
        raise _fail(str(e)) from e
    console.print(f"head is now [bold]{doc.id}[/bold] (pinned to current scorer and gate source)")


def results_table(title: str, results: list) -> Table:
    t = Table(title=title)
    for col, justify in (
        ("workload", "left"),
        ("split", "left"),
        ("cost before", "right"),
        ("cost after", "right"),
        ("improvement", "right"),
        ("idx penalty", "right"),
        ("score", "right"),
        ("violations", "left"),
        ("tokens", "right"),
        ("rounds", "right"),
    ):
        t.add_column(col, justify=justify)
    for r in results:
        bad = bool(r.score.violations)
        t.add_row(
            r.workload_id,
            r.split,
            f"{r.before.workload_cost:,.0f}",
            f"{r.after.workload_cost:,.0f}",
            f"{r.score.improvement:+.1%}",
            f"{r.score.index_penalty:.3f}",
            f"[red]{r.score.score:.3f}[/red]" if bad else f"{r.score.score:.3f}",
            f"[red]{', '.join(r.score.violations)}[/red]" if bad else "-",
            f"{r.tokens:,}",
            str(r.rounds),
        )
    return t


@app.command("eval")
def eval_(
    split: str = typer.Option("train", "--split", "-s", help="train | holdout | canary"),
    version: str = typer.Option(None, "--version", "-v", help="Harness version (default: head)."),
    fake: bool = typer.Option(False, "--fake", help="FakeEnv + fake agent; no DB, no LLM."),
) -> None:
    """Run tune() for every workload of a split with one harness version."""
    from ratchet.harness import runner

    if split not in ("train", "holdout", "canary"):
        raise _fail("--split must be train, holdout or canary")
    try:
        if fake:
            from ratchet.fakes import FakeBackend
            from ratchet.harness.gen0 import gen0_spec

            if version is not None:
                raise _fail("--fake always runs the gen-0 spec; drop --version")
            backend, spec, vid, database = FakeBackend(), gen0_spec(), "h_0000", None
        else:
            from ratchet import db
            from ratchet.harness.store import HarnessStore

            backend = runner.RealBackend()
            database = db.get_db()
            store = HarnessStore(database)
            doc = store.get(version) if version is not None else store.get_head()
            spec, vid = doc.spec, doc.id
    except runner.BackendUnavailable as e:
        raise _fail(str(e)) from e
    except (ConfigError, RuntimeError) as e:
        raise _fail(str(e)) from e

    results = runner.evaluate(backend, spec, vid, split, db=database)  # type: ignore[arg-type]
    if not results:
        raise _fail(f"no {split} workloads found")
    label = " [yellow](FAKE env + agent)[/yellow]" if fake else ""
    console.print(results_table(f"{vid} on {split} · backend={backend.name}{label}", results))
    s = runner.summarise(split, results)  # type: ignore[arg-type]
    console.print(
        f"mean score [bold]{s.mean_score:.3f}[/bold] · mean improvement {s.mean_improvement:+.1%}"
        f" · violations {s.violations} · tokens/workload {s.tokens_per_workload:,.0f}"
    )
