"""Track B CLI: seed, measure, validate (mounted by cli.py)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ratchet.contracts import Measurement, Workload
from ratchet.env import conn
from ratchet.env.detectors import run_detectors
from ratchet.env.measure import measure as run_measure
from ratchet.env.scratch import prepare_scratch
from ratchet.env.validate import (
    WORKLOAD_DIR,
    check_all_static,
    check_live,
    find_workload,
    load_workloads,
)
from ratchet.scoring import query_cost

app = typer.Typer(help="Environment and measurement (Track B)", no_args_is_help=True)
console = Console()


def _measure_one(w: Workload, ns: str) -> tuple[Measurement, list]:
    env = prepare_scratch(w, ns)
    m = run_measure(env, w)
    return m, run_detectors(env, w, m)


def _query_table(w: Workload, m: Measurement) -> Table:
    t = Table(title=f"{w.id} ({w.split}, {w.db}) baseline", title_justify="left")
    for col, just in (
        ("query", "left"),
        ("freq", "right"),
        ("stages", "left"),
        ("nReturned", "right"),
        ("keys", "right"),
        ("docs", "right"),
        ("lookup", "right"),
        ("ratio", "right"),
        ("cost", "right"),
        ("freq×cost", "right"),
    ):
        t.add_column(col, justify=just)
    freqs = {q.id: q.freq for q in w.queries}
    for q in w.queries:
        s = m.per_query[q.id]
        examined = s.keys_examined + s.docs_examined + s.lookup_examined
        ratio = examined / max(1, s.n_returned)
        stages = " → ".join(s.stages)
        if "COLLSCAN" in s.stages:
            stages = stages.replace("COLLSCAN", "[red]COLLSCAN[/]")
        if s.has_in_memory_sort:
            stages = stages.replace("SORT", "[yellow]SORT[/]", 1)
        c = query_cost(s)
        t.add_row(
            q.id,
            f"{freqs[q.id]:g}",
            stages,
            str(s.n_returned),
            f"{s.keys_examined:,}",
            f"{s.docs_examined:,}",
            f"{s.lookup_examined:,}",
            f"[red]{ratio:,.0f}[/]" if ratio > 50 else f"{ratio:,.1f}",
            f"{c:,}",
            f"{freqs[q.id] * c:,.0f}",
        )
    t.caption = f"workload cost Σ freq×cost = {m.workload_cost:,.0f}   indexes: " + "; ".join(
        f"{c}: {', '.join(n for n in names if n != '_id_') or '-'}"
        for c, names in m.indexes.items()
    )
    return t


def _findings_table(findings: list) -> Table:
    t = Table(title="Detector findings", title_justify="left")
    for col in ("query", "detector", "sev", "evidence", "hint"):
        t.add_column(col, overflow="fold")
    colour = {"high": "red", "medium": "yellow", "low": "dim"}
    for f in findings:
        ev = ", ".join(f"{k}={v}" for k, v in f.evidence.items() if v is not None)
        t.add_row(f.query_id, f.detector, f"[{colour[f.severity]}]{f.severity}[/]", ev, f.hint)
    return t


@app.command()
def measure(
    workload: str = typer.Argument(None, help="workload id, e.g. mflix_w01"),
    all_: bool = typer.Option(False, "--all", help="baseline cost table for every workload"),
    ns: str = typer.Option(None, help="scratch namespace (default: RATCHET_NS)"),
) -> None:
    """Before-state explain table + detector findings for one workload (or --all)."""
    ns = ns or conn.current_ns()
    if all_:
        t = Table(title=f"Baseline workload costs (ns={ns})", title_justify="left")
        for col, j in (
            ("workload", "left"),
            ("split", "left"),
            ("db", "left"),
            ("queries", "right"),
            ("Σ freq×cost", "right"),
            ("worst query", "left"),
            ("high", "right"),
            ("findings", "left"),
        ):
            t.add_column(col, justify=j)
        for w in load_workloads(WORKLOAD_DIR):
            m, fs = _measure_one(w, ns)
            worst = max(w.queries, key=lambda q: q.freq * query_cost(m.per_query[q.id]))
            dets = sorted({f.detector for f in fs})
            t.add_row(
                w.id,
                w.split,
                w.db,
                str(len(w.queries)),
                f"{m.workload_cost:,.0f}",
                f"{worst.id} ({query_cost(m.per_query[worst.id]):,})",
                str(sum(f.severity == "high" for f in fs)),
                ", ".join(dets),
            )
        console.print(t)
        return
    if not workload:
        raise typer.BadParameter("give a workload id or --all")
    w = find_workload(workload)
    m, fs = _measure_one(w, ns)
    console.print(_query_table(w, m))
    console.print(_findings_table(fs))


@app.command()
def seed(
    ns: str = typer.Option(None, help="scratch namespace (default: RATCHET_NS)"),
    live: bool = typer.Option(False, help="also run live validation on the source data"),
) -> None:
    """Validate workloads, store them in ratchet_<ns>.workloads, prepare scratch copies."""
    ns = ns or conn.current_ns()
    workloads = load_workloads(WORKLOAD_DIR)
    reports, cross = check_all_static(workloads)
    if live:
        ro = conn.ro_client()
        for w, r in zip(workloads, reports, strict=True):
            check_live(w, ro[w.db], r)
    errors = [f"{r.workload_id}: {e}" for r in reports for e in r.errors] + cross
    if errors:
        for e in errors:
            console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    db = conn.rw_client()[f"ratchet_{ns}"]
    for w in workloads:
        db["workloads"].replace_one({"_id": w.id}, {"_id": w.id, **w.model_dump()}, upsert=True)
        env = prepare_scratch(w, ns)
        copied = ", ".join(f"{c} ({n:,})" for c, n in env.copied.items()) or "already copied"
        console.print(f"[green]✓[/] {w.id:18s} {w.split:8s} scratch: {copied}")
    db["workloads"].create_index("split")
    console.print(
        f"seeded {len(workloads)} workloads into {db.name}.workloads; "
        f"scratch DB {conn.scratch_db_name(ns)} (checksum mode: {env.checksum_mode})"
    )


@app.command()
def validate(
    static: bool = typer.Option(False, "--static", help="no database needed"),
    workload_ids: list[str] = typer.Argument(None),
) -> None:
    """Validate workloads/*.json (determinism, 1-500 results, tie-breaks, splits)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wl_validate", WORKLOAD_DIR / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = (["--static"] if static else []) + list(workload_ids or [])
    raise typer.Exit(mod.main(args))
