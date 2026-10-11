"""Run the tuning agent over a split with one harness version and summarise the results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel
from pymongo.database import Database

from ratchet.config import get_settings, repo_root
from ratchet.contracts import Split, TuneResult, Workload
from ratchet.db import TUNE_RESULTS
from ratchet.harness.models import HarnessSpec

Mode = Literal["normal", "frozen_memory"]


class BackendUnavailable(RuntimeError):
    pass


class Backend(Protocol):
    name: str

    def workloads(self, split: Split | None = None) -> list[Workload]: ...

    def tune(self, workload: Workload, harness: HarnessSpec, mode: Mode) -> TuneResult: ...


def load_workload_files(directory: Path | None = None) -> list[Workload]:
    directory = directory or repo_root() / "workloads"
    return [
        Workload.model_validate(json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))
    ]


class RealBackend:
    """Track B environment + Track C agent, imported lazily so Track A works before they land."""

    name = "atlas"

    def __init__(self, ns: str | None = None) -> None:
        try:
            from ratchet.agent import tune  # type: ignore[attr-defined]
            from ratchet.env import prepare_scratch  # type: ignore[attr-defined]
        except ImportError as e:
            raise BackendUnavailable(
                f"real backend not available yet ({e}); use --fake until Tracks B and C land"
            ) from e
        self._tune, self._prepare = tune, prepare_scratch
        self.ns = ns or get_settings().ns

    def workloads(self, split: Split | None = None) -> list[Workload]:
        return [w for w in load_workload_files() if split is None or w.split == split]

    def tune(self, workload: Workload, harness: HarnessSpec, mode: Mode) -> TuneResult:
        return self._tune(workload, harness, self._prepare(workload, self.ns), mode)


class SplitSummary(BaseModel):
    split: Split
    n: int
    mean_score: float
    mean_improvement: float
    violations: int
    workloads_with_violations: int
    tokens_per_workload: float

    @property
    def pass_rate(self) -> float:
        return 1.0 if self.n == 0 else 1 - self.workloads_with_violations / self.n


def summarise(split: Split, results: list[TuneResult]) -> SplitSummary:
    n = len(results)
    return SplitSummary(
        split=split,
        n=n,
        mean_score=sum(r.score.score for r in results) / n if n else 0.0,
        mean_improvement=sum(r.score.improvement for r in results) / n if n else 0.0,
        violations=sum(len(r.score.violations) for r in results),
        workloads_with_violations=sum(1 for r in results if r.score.violations),
        tokens_per_workload=sum(r.tokens for r in results) / n if n else 0.0,
    )


def save_results(db: Database, results: list[TuneResult]) -> None:
    for r in results:
        db[TUNE_RESULTS].replace_one(
            {"harness_version": r.harness_version, "workload_id": r.workload_id},
            r.model_dump(mode="json"),
            upsert=True,
        )


def load_results(db: Database, version: str, split: Split) -> list[TuneResult]:
    cur = db[TUNE_RESULTS].find({"harness_version": version, "split": split}, {"_id": 0})
    return [TuneResult.model_validate(d) for d in cur.sort("workload_id", 1)]


def evaluate(
    backend: Backend,
    harness: HarnessSpec,
    version: str,
    split: Split,
    db: Database | None = None,
    on_result=None,
) -> list[TuneResult]:
    """tune() every workload of `split`. Memory is frozen for anything that is not train."""
    mode: Mode = "normal" if split == "train" else "frozen_memory"
    results: list[TuneResult] = []
    for w in backend.workloads(split):
        r = backend.tune(w, harness, mode)
        r = r.model_copy(update={"harness_version": version, "split": w.split})
        results.append(r)
        if on_result:
            on_result(r)
    if db is not None:
        save_results(db, results)
    return results
