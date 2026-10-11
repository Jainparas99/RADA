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
