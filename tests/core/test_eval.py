from rich.console import Console
from typer.testing import CliRunner

from ratchet import cli_core
from ratchet import db as dbm
from ratchet.cli import app
from ratchet.contracts import PatchOp
from ratchet.harness import runner
from ratchet.harness.patch import apply_patch
from tests.fakes import FakeBackend


def test_evaluate_runs_each_workload_of_the_split(spec):
    backend = FakeBackend()
    results = runner.evaluate(backend, spec, "h_0000", "train")
    assert [r.workload_id for r in results] == ["fake_train_01", "fake_train_02", "fake_train_03"]
    assert all(r.harness_version == "h_0000" and r.split == "train" for r in results)
    assert all(r.trace and r.tokens > 0 for r in results)
    s = runner.summarise("train", results)
    assert s.n == 3 and s.violations == 0 and 0 < s.mean_score < 1


def test_memory_is_frozen_outside_train(spec):
    backend = FakeBackend()
    for split in ("train", "holdout", "canary"):
        runner.evaluate(backend, spec, "h_0000", split)
    modes = dict(backend.modes)
    assert modes["fake_train_01"] == "normal"
    assert modes["fake_holdout_01"] == modes["fake_canary_01"] == "frozen_memory"


def test_a_better_harness_scores_higher_on_holdout(spec):
    better = apply_patch(
        spec,
        [
            PatchOp(op="replace", path="/context_policy/include_detector_findings", value=True),
            PatchOp(op="add", path="/planner_rules/-", value="Move $match before $lookup."),
        ],
    )
    base = runner.summarise("holdout", runner.evaluate(FakeBackend(), spec, "h_0000", "holdout"))
    new = runner.summarise("holdout", runner.evaluate(FakeBackend(), better, "h_0001", "holdout"))
    assert new.mean_score > base.mean_score + 0.03


def test_fake_canary_passes_at_gen0_and_traps_over_indexing(spec):
    res = runner.evaluate(FakeBackend(), spec, "h_0000", "canary")
    assert res[0].score.violations == []
    assert runner.summarise("canary", res).pass_rate == 1.0
    greedy = apply_patch(
        spec, [PatchOp(op="add", path="/planner_rules/-", value="Index every field you see.")]
    )
    res = runner.evaluate(FakeBackend(), greedy, "h_0001", "canary")
    assert res[0].score.violations == ["INDEX_BUDGET_EXCEEDED"] and res[0].score.score == 0.0
    assert runner.summarise("canary", res).pass_rate == 0.0


def test_a_rule_naming_train_collections_lifts_train_only(spec):
    overfit = apply_patch(
        spec,
        [
            PatchOp(
                op="add",
                path="/planner_rules/-",
                value="Take care with movies, listings, restaurants.",
            )
        ],
    )

    def mean(s, split):
        return runner.summarise(split, runner.evaluate(FakeBackend(), s, "h", split)).mean_score

    assert mean(overfit, "train") >= mean(spec, "train") + 0.10
    assert mean(overfit, "holdout") == mean(spec, "holdout")


def test_results_are_stored_and_reloaded(db, spec):
    results = runner.evaluate(FakeBackend(), spec, "h_0000", "train", db=db)
    runner.evaluate(FakeBackend(), spec, "h_0000", "train", db=db)  # upsert, not duplicate
    assert db[dbm.TUNE_RESULTS].count_documents({}) == 3
    assert runner.load_results(db, "h_0000", "train") == results
    assert runner.load_results(db, "h_0000", "holdout") == []


def test_cli_eval_fake_prints_a_table(monkeypatch):
    monkeypatch.setattr(cli_core, "console", Console(width=200))
    out = CliRunner().invoke(app, ["eval", "--split", "holdout", "--fake"])
    assert out.exit_code == 0, out.output
    assert "fake_holdout_01" in out.output and "FAKE" in out.output and "mean score" in out.output


def test_cli_eval_rejects_bad_split_and_missing_real_backend():
    assert CliRunner().invoke(app, ["eval", "--split", "test", "--fake"]).exit_code == 1
    out = CliRunner().invoke(app, ["eval", "--split", "train"])
    assert out.exit_code == 1 and "--fake" in out.output
