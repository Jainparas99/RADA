"""Validate workloads/*.json (SPEC §5.2).

    uv run python workloads/validate.py            # static checks + live checks on the source data
    uv run python workloads/validate.py --static   # no database needed
    uv run python workloads/validate.py mflix_w01  # only some workloads

Exit code 1 if any workload fails. See ratchet.env.validate for the rules.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from ratchet.env import conn
from ratchet.env.validate import WORKLOAD_DIR, check_all_static, check_live, load_workloads


def main(argv: list[str]) -> int:
    static_only = "--static" in argv
    only = [a for a in argv if not a.startswith("--")]
    workloads = load_workloads(WORKLOAD_DIR)
    reports, cross = check_all_static(workloads)
    pairs = [(w, r) for w, r in zip(workloads, reports, strict=True) if not only or w.id in only]
    console = Console()
    if not static_only:
        ro = conn.ro_client()
        for w, r in pairs:
            check_live(w, ro[w.db], r)
    t = Table(title=f"Workload validation ({'static' if static_only else 'static + live'})")
    for col in ("workload", "split", "db", "queries", "results per query", "status"):
        t.add_column(col)
    for w, r in pairs:
        counts = " ".join(f"{qid}={v['n']}" for qid, v in r.live.items()) or "-"
        t.add_row(
            w.id,
            w.split,
            w.db,
            str(len(w.queries)),
            counts,
            "[green]ok[/]" if r.ok else f"[red]{len(r.errors)} error(s)[/]",
        )
    console.print(t)
    failed = False
    for w, r in pairs:
        for e in r.errors:
            console.print(f"[red]{w.id}[/] {e}")
            failed = True
        for wmsg in r.warnings:
            console.print(f"[yellow]{w.id}[/] {wmsg}")
    for e in cross:
        console.print(f"[red]cross-workload[/] {e}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
