"""Transfer test (SPEC §11.4): does a harness evolved with one model help other models?

For every planner model we can call, run gen 0 and the best harness on the HOLDOUT workloads
(memory frozen) and compare mean scores. Results go to `transfer_runs` and render as a Markdown
table for the README.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ratchet.contracts import TuneResult, Workload

TuneFn = Callable[[Workload, dict, str], TuneResult]  # (workload, harness doc, version) -> result


class HarnessRun(BaseModel):
    version: str
    n: int
    mean_score: float
    mean_improvement: float
    violations: int
    tokens_per_workload: float


class TransferRow(BaseModel):
    model: str
    gen0: HarnessRun
    best: HarnessRun

    @property
    def lift(self) -> float:
        return self.best.mean_score - self.gen0.mean_score


class TransferReport(BaseModel):
    best_version: str
    evolved_with: str
    split: str = "holdout"
    rows: list[TransferRow]
    skipped: dict[str, str] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def with_planner(harness_doc: dict, model: str) -> dict:
    doc = copy.deepcopy(harness_doc)
    doc.setdefault("model_routing", {})["planner"] = model
    return doc


def summarise(version: str, results: list[TuneResult]) -> HarnessRun:
    n = len(results)
    return HarnessRun(
        version=version,
        n=n,
        mean_score=sum(r.score.score for r in results) / n if n else 0.0,
        mean_improvement=sum(r.score.improvement for r in results) / n if n else 0.0,
        violations=sum(len(r.score.violations) for r in results),
        tokens_per_workload=sum(r.tokens for r in results) / n if n else 0.0,
    )


def run_transfer(
    workloads: list[Workload],
    gen0_doc: dict,
    best_doc: dict,
    best_version: str,
    models: list[str],
    tune_fn: TuneFn,
    on_result: Callable[[str, str, TuneResult], None] | None = None,
) -> TransferReport:
    holdout = [w for w in workloads if w.split == "holdout"]
    rows: list[TransferRow] = []
    skipped: dict[str, str] = {}
    for model in models:
        runs: dict[str, list[TuneResult]] = {"gen0": [], "best": []}
        try:
            for label, doc in (("gen0", gen0_doc), ("best", best_doc)):
                for w in holdout:
                    r = tune_fn(w, with_planner(doc, model), label)
                    runs[label].append(r)
                    if on_result:
                        on_result(model, label, r)
        except Exception as e:  # one model failing must not sink the table
            skipped[model] = f"{type(e).__name__}: {e}"[:200]
            continue
        rows.append(
            TransferRow(
                model=model,
                gen0=summarise("gen0", runs["gen0"]),
                best=summarise(best_version, runs["best"]),
            )
        )
    evolved_with = str(best_doc.get("model_routing", {}).get("planner", "?"))
    return TransferReport(
        best_version=best_version, evolved_with=evolved_with, rows=rows, skipped=skipped
    )


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def to_markdown(rep: TransferReport) -> str:
    lines = [
        f"Harness `{rep.best_version}` (evolved with `{rep.evolved_with}`) vs gen 0, "
        f"{rep.split} workloads, memory frozen. Score = improvement − index penalty; "
        "any violation scores 0.",
        "",
        "| planner model | gen 0 score | best score | lift | gen 0 → best improvement "
        "| violations (gen 0 / best) | tokens/workload (best) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rep.rows:
        mark = " (evolved with)" if r.model == rep.evolved_with else ""
        lines.append(
            f"| `{r.model}`{mark} | {r.gen0.mean_score:.2f} | {r.best.mean_score:.2f} "
            f"| {r.lift:+.2f} | {pct(r.gen0.mean_improvement)} → {pct(r.best.mean_improvement)} "
            f"| {r.gen0.violations} / {r.best.violations} | {r.best.tokens_per_workload:,.0f} |"
        )
    for m, why in rep.skipped.items():
        lines.append(f"| `{m}` | skipped: {why} | | | | | |")
    return "\n".join(lines) + "\n"


def callable_models(candidates: list[str], available: Callable[[str], bool]) -> list[str]:
    return [m for m in dict.fromkeys(candidates) if m and available(m)]


def save(db: Any, rep: TransferReport) -> None:
    from ratchet.db import TRANSFER_RUNS

    db[TRANSFER_RUNS].insert_one(
        {**rep.model_dump(mode="json"), "harness_version": rep.best_version}
    )
