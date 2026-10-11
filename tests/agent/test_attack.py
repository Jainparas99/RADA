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
    assert unique.passed is None and "contract strips" in unique.detail


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
