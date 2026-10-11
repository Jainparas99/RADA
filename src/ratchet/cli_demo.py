"""Track C commands: tune, attack, transfer, chart, watch, replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ratchet.config import ConfigError, get_settings
from ratchet.contracts import ExplainSummary, Measurement, TuneResult, Workload

app = typer.Typer(add_completion=False)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]


def safe_stdout() -> None:
    """Never crash on a console that can't encode a character (Windows cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def require_model(*models: str) -> None:
    """Exit with a clear message unless at least one of `models` (in order) is callable."""
    import ratchet.llm as llm

    if not any(llm.available(m) for m in models):
        console.print(
            f"[red]{' / '.join(dict.fromkeys(models))} not callable: missing API key or SDK.[/red]"
            " Set it in .env (see .env.example)."
        )
        raise typer.Exit(1)


# --------------------------------------------------------------------------- loading


def _fakes() -> Any:
    try:
        import ratchet.fakes as fakes  # type: ignore[import-not-found]
    except ImportError:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import tests.fakes as fakes
    return fakes


def load_workload(wid: str, fake: bool = False) -> Workload:
    if fake:
        fakes = _fakes()
        pool = fakes.fake_workloads() if hasattr(fakes, "fake_workloads") else []
        for w in pool:
            if w.id == wid:
                return w
        return fakes.sample_workload(wid=wid)
    for p in sorted((REPO_ROOT / "workloads").glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(doc, dict) and (doc.get("id") == wid or p.stem == wid):
            return Workload.model_validate(doc)
    try:
        from ratchet import db

        doc = db.coll(db.WORKLOADS).find_one({"id": wid}, {"_id": 0})
    except Exception:
        doc = None
    if doc:
        return Workload.model_validate(doc)
    raise typer.BadParameter(f"workload {wid!r} not found in workloads/*.json or Atlas")


def load_harness(version: str | None, use_db: bool) -> tuple[Any, str]:
    """(harness spec dict, version id). Reads Atlas directly; falls back to gen 0 defaults."""
    from ratchet.agent.harness_view import gen0

    if use_db:
        try:
            from ratchet import db

            vid = version
            if vid is None:
                head = db.coll(db.HARNESS_HEAD).find_one({"_id": "head"})
                vid = head.get("version_id") if head else None
            elif not vid.startswith("h_"):
                vid = f"h_{int(vid):04d}"
            if vid:
                doc = db.coll(db.HARNESS_VERSIONS).find_one({"_id": vid})
                if doc:
                    return doc["spec"], vid
                if version is not None:
                    raise typer.BadParameter(f"no harness version {vid}")
        except typer.BadParameter:
            raise
        except Exception as e:
            console.print(f"[yellow]harness store unavailable ({type(e).__name__}); gen 0[/yellow]")
    return gen0().model_dump(mode="json"), "gen0-local"


# --------------------------------------------------------------------------- rendering


def query_cost(s: ExplainSummary) -> int:
    c = s.keys_examined + s.docs_examined + s.lookup_examined
    return c + (s.docs_examined if s.has_in_memory_sort else 0)


def render_findings(r: TuneResult) -> Table:
    t = Table(title="Detector findings")
    for col in ("query", "detector", "severity", "hint"):
        t.add_column(col)
    for f in r.findings:
        t.add_row(f.query_id, f.detector, f.severity, f.hint)
    if not r.findings:
        t.add_row("-", "none", "", "")
    return t


def render_plan(r: TuneResult) -> Table:
    t = Table(title=f"Plan ({len(r.plan.actions)} actions) — {r.plan.summary}")
    for col in ("#", "action", "target", "detail", "why"):
        t.add_column(col, overflow="fold")
    for i, a in enumerate(r.plan.actions, 1):
        if a.type == "create_index" and a.index:
            target = a.collection or ""
            detail = ", ".join(f"{k}:{d}" for k, d in a.index.keys)
        elif a.type == "drop_index":
            target, detail = a.collection or "", a.index_name or ""
        else:
            target = a.query_id or ""
            detail = json.dumps(
                a.new_query.pipeline
                if a.new_query and a.new_query.pipeline
                else (a.new_query.model_dump(exclude_none=True) if a.new_query else {}),
                default=str,
            )[:120]
        t.add_row(str(i), a.type, target, detail, a.rationale[:80])
    return t


def render_critic(r: TuneResult) -> Table | None:
    rows = [e for e in r.trace if e.get("step") == "critic"]
    if not rows:
        return None
    t = Table(title="Critic")
    for col in ("rev", "model", "approve", "issues"):
        t.add_column(col, overflow="fold")
    for e in rows:
        t.add_row(
            str(e.get("revision", 0)),
            str(e.get("model", "")),
            "[green]yes[/green]" if e.get("approve") else "[red]no[/red]",
            "; ".join(e.get("issues", []))[:200],
        )
    return t


def render_before_after(before: Measurement, after: Measurement) -> Table:
    t = Table(title="Before -> after")
    for col in ("query", "stages", "docs examined", "keys examined", "cost"):
        t.add_column(col)
    for qid, b in before.per_query.items():
        a = after.per_query.get(qid, b)
        t.add_row(
            qid,
            f"{'+'.join(b.stages)} -> {'+'.join(a.stages)}",
            f"{b.docs_examined:,} -> {a.docs_examined:,}",
            f"{b.keys_examined:,} -> {a.keys_examined:,}",
            f"{query_cost(b):,} -> {query_cost(a):,}",
        )
    t.add_row(
        "[b]workload[/b]",
        "",
        "",
        "",
        f"[b]{before.workload_cost:,.0f} -> {after.workload_cost:,.0f}[/b]",
    )
    return t


def render_result(r: TuneResult) -> None:
    console.print(render_findings(r))
    console.print(render_plan(r))
    critic = render_critic(r)
    if critic is not None:
        console.print(critic)
    console.print(render_before_after(r.before, r.after))
    s = r.score
    colour = "red" if s.violations else "green"
    console.print(
        f"improvement [b]{s.improvement:+.1%}[/b]  index penalty {s.index_penalty:.3f}  "
        f"score [{colour}][b]{s.score:.3f}[/b][/{colour}]  violations "
        f"[{colour}]{', '.join(s.violations) or 'none'}[/{colour}]"
    )
    models = ", ".join(f"{k}={v}" for k, v in r.models_used.items())
    console.print(f"[dim]rounds {r.rounds} · tokens {r.tokens:,} · {models}[/dim]")


# --------------------------------------------------------------------------- commands


@app.command()
def tune(
    workload: str = typer.Argument(..., help="Workload id (workloads/*.json or Atlas)."),
    version: str | None = typer.Option(None, "--version", "-v", help="Harness version."),
    fake: bool = typer.Option(False, "--fake", help="FakeEnv instead of the Atlas sandbox."),
    as_json: bool = typer.Option(False, "--json", help="Print the TuneResult as JSON."),
) -> None:
    """Run the tuning agent on one workload: findings, plan, critic verdict, before/after."""
    from ratchet.agent import tune as run_tune
    from ratchet.agent.harness_view import view

    safe_stdout()
    try:
        settings = get_settings()
    except ConfigError as e:
        console.print(f"[red]config error:[/red] {e}")
        raise typer.Exit(1) from e
    w = load_workload(workload, fake=fake)
    spec, vid = load_harness(version, use_db=bool(settings.mongodb_uri_rw))
    require_model(view(spec).model_routing.planner, settings.planner_model)
    if fake:
        api = _fakes().FakeEnv()
        env = api.prepare_scratch(w, settings.ns)
    else:
        try:
            import ratchet.env as api  # Track B
        except ImportError as e:
            console.print(f"[red]Track B env not available ({e}); try --fake[/red]")
            raise typer.Exit(1) from e
        env = api.prepare_scratch(w, settings.ns)
    console.print(f"[b]tune[/b] {w.id} ({w.split}) · harness {vid} · ns {settings.ns}")
    r = run_tune(w, spec, env, "normal", api=api, harness_version=vid)
    if as_json:
        console.print_json(r.model_dump_json(exclude={"trace"}))
    else:
        render_result(r)


def all_workloads(fake: bool) -> list[Workload]:
    if fake:
        fakes = _fakes()
        if hasattr(fakes, "fake_workloads"):
            return list(fakes.fake_workloads())
        return [
            fakes.sample_workload(),
            fakes.sample_workload("fake_h01", "holdout", "orders"),
        ]
    out = []
    for p in sorted((REPO_ROOT / "workloads").glob("*.json")):
        try:
            out.append(Workload.model_validate(json.loads(p.read_text(encoding="utf-8"))))
        except ValueError:
            continue
    return out


def _default_attack_workload(fake: bool) -> Workload:
    if fake:
        return _fakes().sample_workload()
    for p in sorted((REPO_ROOT / "workloads").glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("split") == "train" and doc.get("queries"):
            return Workload.model_validate(doc)
    raise typer.BadParameter("no train workload found in workloads/*.json; pass --workload")


def render_attacks(results: list) -> Table:
    t = Table(title="[SEEDED ATTACK] cheats are injected on purpose; each must be rejected")
    for col in ("level", "path", "cheat", "expected", "got", "score", "result"):
        t.add_column(col, overflow="fold")
    for r in results:
        verdict = {True: "[green]REJECTED[/green]", False: "[red]NOT CAUGHT[/red]"}.get(
            r.passed, "[dim]n/a[/dim]"
        )
        t.add_row(
            r.level,
            r.path,
            r.name,
            r.expected,
            ", ".join(r.got) or (r.detail[:60] if r.detail else "-"),
            "-" if r.score is None else f"{r.score:.2f}",
            verdict,
        )
    return t


@app.command()
def attack(
    inner: bool = typer.Option(True, "--inner/--no-inner", help="Plan-level cheats (scorer)."),
    outer: bool = typer.Option(True, "--outer/--no-outer", help="Harness-patch cheats (gate)."),
    workload: str | None = typer.Option(None, "--workload", "-w", help="Workload to attack."),
    fake: bool = typer.Option(False, "--fake", help="FakeEnv instead of the Atlas sandbox."),
) -> None:
    """Run seeded, labelled cheats at both levels and show each one's reason code."""
    from ratchet import attack as atk

    safe_stdout()
    settings = get_settings()
    results: list = []
    if inner:
        w = load_workload(workload, fake=fake) if workload else _default_attack_workload(fake)
        if fake:
            api: Any = _fakes().FakeEnv()
        else:
            try:
                import ratchet.env as api  # Track B
            except ImportError as e:
                console.print(f"[red]Track B env not available ({e}); try --fake[/red]")
                raise typer.Exit(1) from e
        results += atk.run_inner(w, api, lambda: api.prepare_scratch(w, settings.ns))
    if outer:
        results += atk.run_outer(all_workloads(fake))
    console.print(render_attacks(results))
    failed = [r for r in results if r.passed is False]
    if failed:
        console.print(f"[red]{len(failed)} cheat(s) not caught[/red]")
        raise typer.Exit(1)
    console.print(f"[green]all {sum(r.passed is True for r in results)} cheats rejected[/green]")


# --------------------------------------------------------------------------- phase 3 commands


def _state_db() -> Any:
    from ratchet import db

    settings = get_settings()
    if not settings.mongodb_uri_rw:
        console.print("[red]MONGODB_URI_RW is not set (see .env.example)[/red]")
        raise typer.Exit(1)
    return db.get_db()


def _env_backend(fake: bool) -> Any:
    if fake:
        return _fakes().FakeEnv()
    try:
        import ratchet.env as api  # Track B
    except ImportError as e:
        console.print(f"[red]Track B env not available ({e}); try --fake[/red]")
        raise typer.Exit(1) from e
    return api


def _tune_with(api: Any, spec: dict, version: str, mode: str = "frozen_memory"):
    from ratchet.agent import tune as run_tune

    ns = get_settings().ns

    def fn(w: Workload) -> TuneResult:
        return run_tune(w, spec, api.prepare_scratch(w, ns), mode, api=api, harness_version=version)

    return fn


@app.command()
def transfer(
    harness: str | None = typer.Option(None, "--harness", help="Best version (default: head)."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated planner models."),
    fake: bool = typer.Option(False, "--fake", help="FakeEnv instead of the Atlas sandbox."),
    out: Path = typer.Option(Path("docs/transfer.md"), "--out", help="Markdown table output."),
) -> None:
    """Gen 0 vs the best harness on HOLDOUT for every callable planner model (SPEC §11.4)."""
    import ratchet.llm as llm
    from ratchet import transfer as tr

    safe_stdout()
    settings = get_settings()
    use_db = bool(settings.mongodb_uri_rw)
    best_doc, best_v = load_harness(harness, use_db=use_db)
    try:
        gen0_doc, _ = load_harness("0", use_db=use_db) if use_db else load_harness(None, False)
    except typer.BadParameter:
        gen0_doc, _ = load_harness(None, use_db=False)
    candidates = (
        [m.strip() for m in models.split(",")]
        if models
        else [
            *best_doc.get("protected", {}).get("model_allowlist", []),
            settings.planner_model,
            settings.critic_model,
        ]
    )
    usable = tr.callable_models(candidates, llm.available)
    skipped = [m for m in dict.fromkeys(candidates) if m and m not in usable]
    if not usable:
        console.print("[red]no callable planner model (check API keys)[/red]")
        raise typer.Exit(1)
    api = _env_backend(fake)
    workloads = all_workloads(fake)
    ns = settings.ns
    from ratchet.agent import tune as run_tune

    def tune_fn(w: Workload, doc: dict, label: str) -> TuneResult:
        return run_tune(
            w, doc, api.prepare_scratch(w, ns), "frozen_memory", api=api, harness_version=label
        )

    def progress(model: str, label: str, r: TuneResult) -> None:
        console.print(f"[dim]{model} {label} {r.workload_id}: {r.score.score:.3f}[/dim]")

    rep = tr.run_transfer(workloads, gen0_doc, best_doc, best_v, usable, tune_fn, progress)
    rep.skipped.update({m: "no key / SDK" for m in skipped})
    md = tr.to_markdown(rep)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    if use_db:
        try:
            tr.save(_state_db(), rep)
        except Exception as e:
            console.print(f"[yellow]could not store transfer_runs ({type(e).__name__})[/yellow]")
    t = Table(title=f"Transfer: {best_v} vs gen 0 on holdout")
    for col in ("planner model", "gen 0", "best", "lift", "violations g0/best"):
        t.add_column(col)
    for r in rep.rows:
        t.add_row(
            r.model,
            f"{r.gen0.mean_score:.3f}",
            f"{r.best.mean_score:.3f}",
            f"{r.lift:+.3f}",
            f"{r.gen0.violations}/{r.best.violations}",
        )
    console.print(t)
    console.print(f"markdown written to {out}")


@app.command()
def chart(
    out: Path = typer.Option(Path("docs/chart.png"), "--out", help="PNG path."),
) -> None:
    """PNG: train/holdout score per generation, accepted vs rejected candidates."""
    from ratchet.viz.chart import load_points, render_chart

    safe_stdout()
    points = load_points(_state_db())
    if not points:
        console.print("[yellow]no generations recorded yet; run `ratchet evolve`[/yellow]")
        raise typer.Exit(1)
    path = render_chart(points, out)
    acc = sum(p.verdict == "ACCEPT" for p in points if p.gen > 0)
    rej = sum(p.verdict == "REJECT" for p in points)
    console.print(f"wrote {path} ({len(points)} points, {acc} accepted, {rej} rejected)")


@app.command()
def watch(
    interval: float = typer.Option(2.0, "--interval", help="Polling interval (fallback)."),
    poll: bool = typer.Option(False, "--poll", help="Skip change streams; poll."),
) -> None:
    """Live tail of harness versions, gate decisions and generations."""
    from ratchet.viz.watch import watch as run_watch

    safe_stdout()
    try:
        run_watch(_state_db(), console.print, interval=interval, use_change_streams=not poll)
    except KeyboardInterrupt:
        console.print("[dim]stopped[/dim]")


@app.command()
def replay(
    version: str | None = typer.Option(None, "--version", "-v", help="Harness version."),
    gen: int | None = typer.Option(None, "--gen", help="Replay this generation's candidate."),
    split: str = typer.Option("train", "--split", help="train | holdout | canary"),
    fake: bool = typer.Option(False, "--fake", help="FakeEnv instead of the Atlas sandbox."),
) -> None:
    """Re-run a recorded generation from llm_cache only (no provider calls)."""
    from ratchet import replay as rp

    safe_stdout()
    db = _state_db()
    try:
        vid = rp.resolve_version(db, version, gen)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    spec, vid = load_harness(vid, use_db=True)
    api = _env_backend(fake)
    workloads = [w for w in all_workloads(fake) if w.split == split]
    from ratchet.db import TUNE_RESULTS

    recorded = {
        d["workload_id"]: d["score"]["score"]
        for d in db[TUNE_RESULTS].find({"harness_version": vid, "split": split})
    }
    mode = "normal" if split == "train" else "frozen_memory"
    rows = rp.replay(workloads, _tune_with(api, spec, vid, mode), recorded)
    t = Table(title=f"Replay {vid} on {split} from llm_cache (no provider calls)")
    for col in ("workload", "score", "recorded", "violations", "llm calls", "cache misses"):
        t.add_column(col)
    for r in rows:
        rec = "-" if r.recorded_score is None else f"{r.recorded_score:.3f}"
        miss = f"[red]{r.cache_misses}[/red]" if r.cache_misses else "0"
        t.add_row(
            r.workload_id,
            f"{r.score:.3f}",
            rec,
            ", ".join(r.violations) or "-",
            str(r.llm_calls),
            miss,
        )
    console.print(t)
