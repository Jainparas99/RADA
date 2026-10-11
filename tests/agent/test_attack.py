import pytest
from typer.testing import CliRunner

from ratchet import attack as atk
from ratchet.cli import app
from tests.fakes import FakeEnv, sample_workload


@pytest.fixture
def results():
    env, w = FakeEnv(), sample_workload()
    return atk.run_inner(w, env, lambda: env.prepare_scratch(w))


def test_every_inner_cheat_is_rejected_with_its_code(results):
    by = {(r.name, r.path): r for r in results}
    assert {n for n, _ in by} == {
        "rewrite_changes_filter",
        "index_every_field",
        "out_precomputed_answers",
        "unique_index",
    }
    for r in results:
        assert r.passed is not False, r
    for name, code in [
        ("rewrite_changes_filter", "RESULT_MISMATCH"),
        ("index_every_field", "INDEX_BUDGET_EXCEEDED"),
        ("out_precomputed_answers", "FORBIDDEN_ACTION"),
        ("unique_index", "FORBIDDEN_ACTION"),
    ]:
        agent = by[(name, "agent")]
        assert agent.passed is True and code in agent.got and agent.score == 0.0


def test_scorer_path_catches_expressible_cheats_without_the_agent(results):
    by = {(r.name, r.path): r for r in results}
    assert by[("rewrite_changes_filter", "scorer")].got == ["RESULT_MISMATCH"]
    assert "INDEX_BUDGET_EXCEEDED" in by[("index_every_field", "scorer")].got
    assert "FORBIDDEN_ACTION" in by[("out_precomputed_answers", "scorer")].got
    unique = by[("unique_index", "scorer")]
    assert unique.passed is True and "FORBIDDEN_ACTION" in unique.got


def test_filter_cheat_follows_spec_example():
    rw = atk.filter_rewrite(sample_workload())
    assert rw["new_query"]["filter"]["year"] == {"$gt": 2015}  # year > 2000 -> 2015


def test_cheats_are_labelled():
    for c in atk.inner_cheats(sample_workload()):
        assert atk.LABEL in c.plan["summary"]
        assert all(atk.LABEL in a["rationale"] for a in c.plan["actions"])


def test_a_cheat_that_slips_through_is_reported_as_failure():
    class Blind(FakeEnv):
        def score(self, workload, before, after, plan, apply):
            rep = super().score(workload, before, after, plan, apply)
            return rep.model_copy(update={"violations": [], "score": 0.9})

    env, w = Blind(), sample_workload()
    res = atk.run_inner(w, env, lambda: env.prepare_scratch(w))
    assert any(r.passed is False for r in res)


def test_cli_attack_fake_inner():
    res = CliRunner().invoke(app, ["attack", "--fake", "--no-outer"])
    assert res.exit_code == 0, res.output
    assert "SEEDED ATTACK" in res.output and "REJECTED" in res.output
    assert "NOT CAUGHT" not in res.output


# ---------------------------------------------------------------- outer


def _workloads():
    return [sample_workload(), sample_workload("fake_h01", "holdout", "orders")]


def test_outer_cheats_target_the_right_fields():
    cheats = {c.name: c for c in atk.outer_cheats(_workloads())}
    assert cheats["raise_token_budget"].patch[0]["path"].startswith("/protected/")
    assert cheats["extend_model_allowlist"].patch[0]["path"].startswith("/protected/")
    leak = cheats["holdout_leak_rule"].patch[0]["value"]
    assert "orders" in leak and "genres" in leak  # names from the HOLDOUT workload
    assert cheats["route_to_unlisted_model"].patch[0]["path"] == "/model_routing/planner"
    od = cheats["overfit_train"]
    assert od.cand_metrics["train"] > od.parent_metrics["train"]
    assert od.cand_metrics["holdout"] <= od.parent_metrics["holdout"]


def test_outer_reports_na_when_gate_not_merged(monkeypatch):
    def boom(_):
        raise ImportError("x", name="ratchet.gate.static")

    monkeypatch.setattr(atk, "GateAdapter", boom)
    res = atk.run_outer(_workloads())
    assert len(res) == 5 and all(r.passed is None for r in res)


class StubGate:
    def __init__(self, codes):
        self.codes = codes

    def static(self, patch):
        return self.codes

    def dynamic(self, parent, cand):
        return self.codes


def test_outer_pass_requires_the_expected_code():
    ok = atk.run_outer(_workloads(), gate=StubGate(["NO_HOLDOUT_GAIN"]))
    assert all(r.passed is False for r in ok)  # wrong code is a failure, not a pass


def test_outer_against_track_a_gate():
    pytest.importorskip("ratchet.gate.static")
    pytest.importorskip("ratchet.harness.gen0")
    res = atk.run_outer(_workloads())
    bad = [(r.name, r.expected, r.got) for r in res if r.passed is not True]
    assert not bad, bad
