"""`ratchet` entry point. Mounts one Typer app per track so CLI edits never conflict."""

from __future__ import annotations

import typer

from ratchet import cli_core, cli_demo, cli_env

app = typer.Typer(
    name="ratchet",
    help="Ratchet: a self-improving harness for MongoDB query tuning.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Ratchet: a self-improving harness for MongoDB query tuning."""


# Commands from every track live at the top level: `ratchet init`, `ratchet seed`, `ratchet tune`.
for _sub in (cli_core.app, cli_env.app, cli_demo.app):
    app.registered_commands.extend(_sub.registered_commands)

if __name__ == "__main__":
    app()
