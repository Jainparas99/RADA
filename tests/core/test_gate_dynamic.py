"""Every dynamic reason code has a test that triggers it and one that doesn't."""

import pytest

from ratchet.gate.dynamic import RunMetrics, check_dynamic, metrics_from_results
from ratchet.harness import runner
from tests.fakes import FakeBackend


def M(**kw):  # noqa: N802
    base = dict(
        train=0.50, holdout=0.50, violations=0, canary_violations=0, canary_n=3,
        tokens_per_workload=1000.0,
    )  # fmt: skip
    return RunMetrics(**{**base, **kw})


PARENT = M()


def test_clear_improvement_is_accepted():
    assert check_dynamic(PARENT, M(train=0.58, holdout=0.56)) == []


# CANARY_FAILED
def test_canary_failed():
    assert check_dynamic(PARENT, M(holdout=0.9, canary_violations=1)) == ["CANARY_FAILED"]
    assert "CANARY_FAILED" not in check_dynamic(PARENT, M(holdout=0.9, canary_violations=0))


# INNER_VIOLATIONS_UP
def test_inner_violations_up():
    assert check_dynamic(PARENT, M(holdout=0.9, violations=1)) == ["INNER_VIOLATIONS_UP"]
    parent = M(violations=2)
    assert check_dynamic(parent, M(holdout=0.9, violations=2)) == []
    assert check_dynamic(parent, M(holdout=0.9, violations=1)) == []


# NO_HOLDOUT_GAIN
@pytest.mark.parametrize("holdout", [0.0, 0.40, 0.50, 0.52, 0.5299])
def test_no_holdout_gain(holdout):
    assert check_dynamic(PARENT, M(holdout=holdout)) == ["NO_HOLDOUT_GAIN"]


@pytest.mark.parametrize("holdout", [0.53, 0.60, 1.0])
def test_holdout_gain_at_or_over_threshold(holdout):
    assert check_dynamic(PARENT, M(holdout=holdout)) == []


def test_equal_holdout_is_accepted_only_when_meaningfully_cheaper():
    assert check_dynamic(PARENT, M(holdout=0.50, tokens_per_workload=850)) == []
    assert check_dynamic(PARENT, M(holdout=0.49, tokens_per_workload=800)) == []
    assert check_dynamic(PARENT, M(holdout=0.51, tokens_per_workload=700)) == []
    # not cheap enough
    assert check_dynamic(PARENT, M(holdout=0.50, tokens_per_workload=860)) == ["NO_HOLDOUT_GAIN"]
    # cheaper but holdout fell by more than the tolerance
    assert check_dynamic(PARENT, M(holdout=0.48, tokens_per_workload=500)) == ["NO_HOLDOUT_GAIN"]


# OVERFIT_DIVERGENCE
def test_overfit_divergence():
    assert check_dynamic(PARENT, M(train=0.60, holdout=0.50)) == [
        "NO_HOLDOUT_GAIN",
        "OVERFIT_DIVERGENCE",
    ]
    assert "OVERFIT_DIVERGENCE" in check_dynamic(PARENT, M(train=0.95, holdout=0.30))
    # cheaper-and-equal does not excuse a train-only jump
    assert check_dynamic(PARENT, M(train=0.70, holdout=0.50, tokens_per_workload=500)) == [
        "OVERFIT_DIVERGENCE"
    ]


def test_train_rise_with_holdout_rise_is_not_overfit():
    assert check_dynamic(PARENT, M(train=0.90, holdout=0.55)) == []
    assert "OVERFIT_DIVERGENCE" not in check_dynamic(PARENT, M(train=0.59, holdout=0.50))


# TOKEN_BUDGET_EXCEEDED
def test_token_budget_exceeded():
    assert check_dynamic(PARENT, M(holdout=0.9, tokens_per_workload=1251)) == [
        "TOKEN_BUDGET_EXCEEDED"
    ]
    assert check_dynamic(PARENT, M(holdout=0.9, tokens_per_workload=1250)) == []
    assert check_dynamic(
        PARENT, M(holdout=0.9, tokens_per_workload=1100), token_budget_per_workload=1000
    ) == ["TOKEN_BUDGET_EXCEEDED"]
    assert (
        check_dynamic(
            PARENT, M(holdout=0.9, tokens_per_workload=1100), token_budget_per_workload=30000
        )
        == []
    )


def test_thresholds_come_from_config():
    strict = {"dynamic": {"holdout_min_gain": 0.2, "token_budget_factor": 1.0}}
    assert check_dynamic(PARENT, M(holdout=0.6, tokens_per_workload=1001), cfg=strict) == [
        "NO_HOLDOUT_GAIN",
        "TOKEN_BUDGET_EXCEEDED",
    ]


def test_everything_at_once_is_reported_in_canonical_order():
    worst = M(train=0.9, holdout=0.1, violations=3, canary_violations=2, tokens_per_workload=9999)
    assert check_dynamic(PARENT, worst) == [
        "CANARY_FAILED",
        "INNER_VIOLATIONS_UP",
        "NO_HOLDOUT_GAIN",
        "OVERFIT_DIVERGENCE",
        "TOKEN_BUDGET_EXCEEDED",
    ]


def test_metrics_from_results(spec):
    b = FakeBackend()
    runs = {s: runner.evaluate(b, spec, "h_0000", s) for s in ("train", "holdout", "canary")}
    m = metrics_from_results(runs["train"], runs["holdout"], runs["canary"])
    assert m.train == pytest.approx(runner.summarise("train", runs["train"]).mean_score)
    assert m.violations == 0 and m.canary_violations == 0 and m.canary_n == 1
    assert m.canary_pass == 1.0 and m.tokens_per_workload > 0
    assert check_dynamic(m, m) == ["NO_HOLDOUT_GAIN"]
