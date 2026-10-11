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
        ("GROQ_API_KEY", s.groq_api_key),
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


# --------------------------------------------------------------------------- outer loop


def _num(v: float | None, fmt: str = ".3f") -> str:
    return "-" if v is None else format(v, fmt)


def history_table(store) -> Table:
    head = store.head_id()
    t = Table(title=f"harness history · head = {head}")
    for col in ("version", "status", "parent", "by", "train", "holdout", "canary", "tok/wl"):
        t.add_column(col, justify="right" if col in ("train", "holdout", "tok/wl") else "left")
    t.add_column("reason codes / change")
    colours = {"accepted": "green", "rejected": "red", "candidate": "yellow"}
    for d in store.history():
        change = ", ".join(f"{p.op} {p.path}" for p in d.patch) or d.rationale.split("\n")[0][:60]
        what = f"[red]{' '.join(d.reason_codes)}[/red]  {change}" if d.reason_codes else change
        t.add_row(
            f"[bold]{d.id}[/bold] ◀" if d.id == head else d.id,
            f"[{colours[d.status]}]{d.status}[/{colours[d.status]}]",
            d.parent or "-",
            d.created_by,
            _num(d.scores.train),
            _num(d.scores.holdout),
            _num(d.scores.canary_pass, ".0%"),
            _num(d.scores.tokens_per_workload, ",.0f"),
            what,
        )
    return t


def print_diff(store, a: str, b: str) -> None:
    import difflib
    import json

    da, db_ = store.get(a), store.get(b)
    console.print(f"[bold]{da.id}[/bold] ({da.status}) → [bold]{db_.id}[/bold] ({db_.status})")
    if db_.parent == da.id and db_.patch:
        console.print("[dim]patch:[/dim]")
        for p in db_.patch:
            console.print(f"  {p.op} {p.path} " + ("" if p.op == "remove" else json.dumps(p.value)))
        if db_.reason_codes:
            console.print(
                f"[red]rejected: {' '.join(db_.reason_codes)}[/red] (spec left unchanged)"
            )
        console.print(f"[dim]why:[/dim] {db_.rationale}")
    left = json.dumps(da.spec.to_doc(), indent=2, sort_keys=True).splitlines()
    right = json.dumps(db_.spec.to_doc(), indent=2, sort_keys=True).splitlines()
    lines = list(difflib.unified_diff(left, right, da.id, db_.id, lineterm="", n=2))
    if not lines:
        console.print("[dim]specs are identical[/dim]")
    for line in lines:
        colour = "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
        console.print(f"[{colour}]{line}[/{colour}]", highlight=False, markup=True)


def _store():
    from ratchet import db
    from ratchet.harness.store import HarnessStore

    return HarnessStore(db.get_db())


def _print_event(kind: str, data: dict) -> None:
    if kind == "gen_start":
        console.rule(f"generation {data['gen']}")
    elif kind == "run":
        console.print(f"  running {data['version']} on {data['split']}…")
    elif kind == "canary_failed":
        console.print(f"  [red]canary failed for {data['version']}[/red]")
    elif kind == "rollback":
        console.print(f"  [yellow]auto-rollback {data['from']} → {data['to']}[/yellow]")
    elif kind == "lessons":
        console.print(
            f"  memory: {data['written']} new lesson(s) from {data['version']} train runs"
        )
    elif kind == "proposal":
        for p in data["patch"]:
            console.print(f"  proposed: {p['op']} {p['path']}")
    elif kind == "gen_end":
        colour = {"ACCEPT": "green", "REJECT": "red"}.get(data["verdict"], "yellow")
        codes = " ".join(data["reason_codes"])
        pm, cm = data.get("parent_metrics") or {}, data.get("candidate_metrics") or {}
        nums = f" holdout {pm['holdout']:.3f} → {cm['holdout']:.3f}" if cm else ""
        console.print(
            f"  [{colour}]{data['verdict']}[/{colour}] {data.get('candidate') or ''} {codes}{nums}"
            f"  (head {data['head_after']}, {data['duration_s']}s)"
        )


@app.command()
def evolve(
    gens: int = typer.Option(1, "--gens", "-g", min=1, help="Generations to run."),
    fake: bool = typer.Option(
        False, "--fake", help="In-memory DB, FakeEnv, fake agent and a SCRIPTED proposer."
    ),
) -> None:
    """Outer loop: propose a harness patch, gate it, promote or reject."""
    from ratchet.evolve import Evolver
    from ratchet.harness import runner

    try:
        if fake:
            import mongomock

            from ratchet.fakes import FakeBackend, FakeEnv, fake_memory_llm, scripted_proposer_llm
            from ratchet.harness.init_db import init_database
            from ratchet.memory import MemoryStore
            from ratchet.proposer.propose import propose

            database = mongomock.MongoClient(tz_aware=True)["ratchet_fake"]
            init_database(database, vector=False)
            llm = scripted_proposer_llm()
            mem_llm = fake_memory_llm()
            mem = MemoryStore(
                database,
                embed=mem_llm.embed,
                complete=mem_llm.complete,
                shape_fn=FakeEnv().query_shape,
                model="fake:scripted",
            )
            console.print(
                "[yellow]FAKE run: in-memory DB, FakeEnv, fake agent, scripted proposer. "
                "Numbers are not real.[/yellow]"
            )
            ev = Evolver(
                database,
                FakeBackend(),
                lambda ctx: propose(ctx, complete=llm.complete, model="fake:scripted"),
                memory=mem,
                proposer_model="fake:scripted",
                on_event=_print_event,
            )
        else:
            from ratchet import db, memory
            from ratchet.proposer.propose import propose

            model = get_settings().proposer_model
            mem = memory.default_store()  # the same store the agent recalls from
            ev = Evolver(
                db.get_db(),
                runner.RealBackend(),
                propose,
                memory=mem,
                proposer_model=model,
                on_event=_print_event,
            )
        records = ev.run(gens)
    except (ConfigError, runner.BackendUnavailable, RuntimeError, ImportError) as e:
        raise _fail(str(e)) from e
    console.print(history_table(ev.store))
    console.print(memory_table(mem))
    if fake:
        accepted = [r for r in records if r.verdict == "ACCEPT"]
        if accepted:
            console.rule("diff of the last accepted patch")
            print_diff(ev.store, accepted[-1].head_before, accepted[-1].candidate)


@app.command()
def history() -> None:
    """Harness versions with verdicts, scores and reason codes."""
    from ratchet.harness.store import StoreError

    try:
        store = _store()
        if store.head_id() is None:
            raise StoreError("no harness head; run `ratchet init`")
        console.print(history_table(store))
    except (ConfigError, StoreError) as e:
        raise _fail(str(e)) from e


@app.command()
def diff(a: str = typer.Argument(...), b: str = typer.Argument(...)) -> None:
    """Coloured spec diff between two harness versions (e.g. `ratchet diff 3 4`)."""
    from ratchet.harness.store import StoreError

    try:
        print_diff(_store(), a, b)
    except (ConfigError, StoreError, ValueError) as e:
        raise _fail(str(e)) from e


@app.command()
def rollback(version: str = typer.Argument(..., help="Accepted version to make the head.")) -> None:
    """Point the head back at an earlier accepted version."""
    from ratchet.harness.store import StoreError

    try:
        doc = _store().rollback(version)
    except (ConfigError, StoreError, ValueError) as e:
        raise _fail(str(e)) from e
    console.print(f"head is now [bold]{doc.id}[/bold]")


def memory_table(store) -> Table:
    stats = store.stats()
    t = Table(
        title=f"lesson memory · {stats['active']} active · uses {stats['uses']}"
        f" · helpful {stats['helpful']} · harmful {stats['harmful']}"
    )
    for col in ("id", "status", "source", "from", "support", "uses", "helpful", "harmful"):
        right = col in ("support", "uses", "helpful", "harmful")
        t.add_column(col, justify="right" if right else "left")
    t.add_column("lesson")
    t.add_column("learned on shape", overflow="fold")
    colours = {"active": "green", "invalidated": "red", "merged": "dim", "evicted": "dim"}
    for x in store.all():
        t.add_row(
            x.id,
            f"[{colours[x.status]}]{x.status}[/{colours[x.status]}]",
            x.source.removeprefix("train_"),
            x.source_workload,
            str(x.support),
            str(x.uses),
            str(x.helpful),
            str(x.harmful),
            x.text,
            x.shape,
        )
    return t


def _memory_store():
    from ratchet import db
    from ratchet.memory import MemoryStore

    return MemoryStore(db.get_db(), embed=lambda texts, **kw: [])  # read-only listing


@app.command()
def memory() -> None:
    """Lessons learned on train workloads, with helpful/harmful counts."""
    try:
        console.print(memory_table(_memory_store()))
    except ConfigError as e:
        raise _fail(str(e)) from e
