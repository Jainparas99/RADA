"""Dynamic gate: compare a candidate's measured results with its parent's (SPEC §9).

Deterministic arithmetic over `RunMetrics`; thresholds come from `config/gate.yaml`.
"""

from __future__ import annotations

from pydantic import BaseModel

from ratchet.config import load_yaml
from ratchet.contracts import TuneResult

DYNAMIC_CODES = [
    "CANARY_FAILED",
    "INNER_VIOLATIONS_UP",
    "NO_HOLDOUT_GAIN",
    "OVERFIT_DIVERGENCE",
    "TOKEN_BUDGET_EXCEEDED",
]

_EPS = 1e-9  # scores are sums of floats; do not let 0.30000000000000004 decide a promotion


class RunMetrics(BaseModel):
    """What the gate needs from one harness version's run."""

    train: float  # mean score
    holdout: float  # mean score
    violations: int  # inner violations on train + holdout
    canary_violations: int  # canary workloads with any inner violation
    canary_n: int = 0
    tokens_per_workload: float  # over train + holdout

    @property
    def canary_pass(self) -> float:
        return 1.0 if self.canary_n == 0 else 1 - self.canary_violations / self.canary_n


def metrics_from_results(
    train: list[TuneResult], holdout: list[TuneResult], canary: list[TuneResult]
) -> RunMetrics:
    def mean(rs: list[TuneResult]) -> float:
        return sum(r.score.score for r in rs) / len(rs) if rs else 0.0

    scored = [*train, *holdout]
    return RunMetrics(
        train=mean(train),
        holdout=mean(holdout),
        violations=sum(len(r.score.violations) for r in scored),
        canary_violations=sum(1 for r in canary if r.score.violations),
        canary_n=len(canary),
        tokens_per_workload=sum(r.tokens for r in scored) / len(scored) if scored else 0.0,
    )


def check_dynamic(
    parent: RunMetrics,
    cand: RunMetrics,
    cfg: dict | None = None,
    token_budget_per_workload: int | None = None,
) -> list[str]:
    """Reason codes for a candidate run against its parent's; empty means ACCEPT."""
    c = (cfg if cfg is not None else load_yaml("gate.yaml")).get("dynamic", {})
    min_gain = c.get("holdout_min_gain", 0.03)
    equal_tol = c.get("holdout_equal_tolerance", 0.01)
    token_drop = c.get("token_drop_for_equal", 0.15)
    overfit_rise = c.get("overfit_train_rise", 0.10)
    token_factor = c.get("token_budget_factor", 1.25)
    codes: set[str] = set()

    if cand.canary_violations > 0:
        codes.add("CANARY_FAILED")

    if cand.violations > parent.violations:
        codes.add("INNER_VIOLATIONS_UP")

    gain = cand.holdout - parent.holdout
    if gain < min_gain - _EPS:
        cheaper = (
            parent.tokens_per_workload > 0
            and cand.tokens_per_workload <= parent.tokens_per_workload * (1 - token_drop) + _EPS
        )
        if not (abs(gain) <= equal_tol + _EPS and cheaper):
            codes.add("NO_HOLDOUT_GAIN")

    if cand.train - parent.train >= overfit_rise - _EPS and cand.holdout <= parent.holdout + _EPS:
        codes.add("OVERFIT_DIVERGENCE")

    if parent.tokens_per_workload > 0 and (
        cand.tokens_per_workload > parent.tokens_per_workload * token_factor + _EPS
    ):
        codes.add("TOKEN_BUDGET_EXCEEDED")
    if (
        token_budget_per_workload is not None
        and cand.tokens_per_workload > token_budget_per_workload
    ):
        codes.add("TOKEN_BUDGET_EXCEEDED")

    return [code for code in DYNAMIC_CODES if code in codes]
