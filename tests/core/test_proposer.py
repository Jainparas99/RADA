import json

import pytest

from ratchet.contracts import GateDecision, PatchOp, ProposerOutput
from ratchet.harness import runner
from ratchet.harness.models import HarnessDoc
from ratchet.proposer.context import (
    ContextLeakError,
    build_context,
    gate_feedback,
    redact,
    select_failures,
)
from ratchet.proposer.propose import SYSTEM, ProposerError, build_messages, parse_output, propose
from tests.fakes import SCRIPTED_PROPOSALS, FakeBackend, FakeLLM


@pytest.fixture
def head(spec):
    return HarnessDoc(id="h_0000", version=0, status="accepted", spec=spec)


@pytest.fixture
def runs(spec):
    b = FakeBackend()
    return {s: runner.evaluate(b, spec, "h_0000", s) for s in ("train", "holdout", "canary")}


def test_context_contains_spec_bounds_failures_and_limits(head, runs):
    ctx = build_context(head, runs["train"], memory_stats={"lessons": 0})
    assert ctx.head_version == "h_0000" and ctx.spec == head.spec.to_doc()
    assert "/planner_rules" in ctx.editable_paths and "/protected" not in ctx.editable_paths
    assert ctx.bounds["numeric"]["/memory_policy/retrieve/k"] == [0, 8]
    assert ctx.patch_limits == {"max_ops": 4, "max_added_chars": 1200}
    assert ctx.train_summary["workloads"] == 3 and ctx.memory_stats == {"lessons": 0}
    assert 1 <= len(ctx.failures) <= 8
    f = ctx.failures[0]
    assert "COLLSCAN_LARGE" in f.detectors and "plan" in f.trace and f.plan["actions"]


@pytest.mark.parametrize("split", ["holdout", "canary"])
def test_context_refuses_non_train_results(head, runs, split):
    with pytest.raises(ContextLeakError, match=split):
        build_context(head, [*runs["train"], runs[split][0]])


def test_select_failures_prefers_distinct_problems_and_caps_at_8(runs):
    base = runs["train"][0]

    def variant(i, score, violations=(), detectors=()):
        findings = [base.findings[0].model_copy(update={"detector": d}) for d in detectors]
        sc = base.score.model_copy(update={"score": score, "violations": list(violations)})
        return base.model_copy(
            update={"workload_id": f"w{i:02d}", "score": sc, "findings": findings}
        )

    pool = [variant(i, 0.1 + i / 100, detectors=["COLLSCAN_LARGE"]) for i in range(10)]
    pool.append(variant(90, 0.0, violations=["RESULT_MISMATCH"]))
    pool.append(variant(91, 0.45, detectors=["LATE_MATCH"]))
    pool.append(variant(92, 0.95, detectors=["UNANCHORED_REGEX"]))  # good score: not a failure
    chosen = select_failures(pool)
    ids = [r.workload_id for r in chosen]
    assert len(chosen) == 8 and len(set(ids)) == 8
    assert ids[:3] == ["w90", "w00", "w91"]  # each adds a new violation/detector
    assert "w92" not in ids


def test_gate_feedback_carries_codes_but_no_holdout_numbers():
    d = GateDecision(
        candidate="h_0003",
        parent="h_0002",
        verdict="REJECT",
        reason_codes=["NO_HOLDOUT_GAIN"],
        metrics={
            "parent": {"train": 0.5, "holdout": 0.123456, "canary_violations": 0},
            "candidate": {"train": 0.7, "holdout": 0.654321, "canary_violations": 1},
        },
    )
    fb = gate_feedback(d, [PatchOp(op="add", path="/planner_rules/-", value="x")])
    dumped = fb.model_dump_json()
    assert fb.reason_codes == ["NO_HOLDOUT_GAIN"] and (fb.train_before, fb.train_after) == (
        0.5,
        0.7,
    )
    assert "0.123456" not in dumped and "0.654321" not in dumped and "canary" not in dumped


def test_redact_scrubs_held_out_names(leak_terms):
    clean, hits = redact(
        {
            "note": "looks like the Sales collection",
            "sold_at": [1, "index ѕаlеѕ now"],
            "ok": "movies",
        },
        leak_terms,
    )
    assert hits >= 3 and not leak_terms.find(json.dumps(clean))
    assert clean["ok"] == "movies"


def test_build_context_redacts_free_text(head, runs, leak_terms):
    dirty = runs["train"][0].model_copy(
        update={"self_diagnostics": [{"category": "tool_error", "note": "saw accounts.balance"}]}
    )
    ctx = build_context(head, [dirty, *runs["train"][1:]], leak_terms=leak_terms)
    assert ctx.redactions >= 1
    assert not leak_terms.find(ctx.model_dump_json())


# ------------------------------------------------------------------------------- propose


@pytest.fixture
def ctx(head, runs):
    return build_context(head, runs["train"])


def test_propose_returns_validated_output(ctx):
    llm = FakeLLM({"proposer": [SCRIPTED_PROPOSALS[1]]})
    usage: dict = {}
    out = propose(ctx, complete=llm.complete, model="fake:m", usage=usage)
    assert isinstance(out, ProposerOutput)
    assert out.patch == [
        PatchOp(op="replace", path="/context_policy/include_detector_findings", value=True)
    ]
    call = llm.calls[0]
    assert (call["role"], call["model"]) == ("proposer", "fake:m")
    assert call["json_schema"] == ProposerOutput.model_json_schema()
    assert call["messages"][0] == {"role": "system", "content": SYSTEM}
    assert (
        usage == {"tokens": usage["tokens"], "calls": 1, "model": "fake:m"} and usage["tokens"] > 0
    )


def test_propose_repairs_once(ctx):
    llm = FakeLLM({"proposer": ["Sure! Here is my patch: {oops", SCRIPTED_PROPOSALS[0]]})
    out = propose(ctx, complete=llm.complete, model="fake:m")
    assert out.patch[0].path == "/planner_rules/-"
    assert len(llm.calls) == 2
    repair = llm.calls[1]["messages"]
    assert repair[-2]["role"] == "assistant" and "not usable" in repair[-1]["content"]


def test_propose_gives_up_after_one_repair(ctx):
    llm = FakeLLM(
        {"proposer": ["nope", {"diagnosis": "d", "patch": [{"op": "move", "path": "/x"}]}]}
    )
    with pytest.raises(ProposerError, match="schema"):
        propose(ctx, complete=llm.complete, model="fake:m")
    assert len(llm.calls) == 2


def test_parse_output_accepts_fenced_json_text():
    class Resp:
        text = "```json\n" + json.dumps(SCRIPTED_PROPOSALS[5]) + "\n```"

    assert parse_output(Resp()).patch[0].op == "add"


def test_prompt_is_the_context_and_nothing_else(ctx):
    messages = build_messages(ctx)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert json.loads(messages[1]["content"])["head_version"] == "h_0000"


def test_llm_layer_errors_become_proposer_errors(ctx):
    class LLMError(RuntimeError):  # same name as ratchet.llm.types.LLMError
        pass

    class LLMJSONError(LLMError):
        pass

    def broken(*a, **kw):
        raise LLMJSONError("model never produced valid JSON")

    with pytest.raises(ProposerError, match="LLMJSONError"):
        propose(ctx, complete=broken, model="fake:m")

    def bug(*a, **kw):
        raise KeyError("a real bug must not be swallowed")

    with pytest.raises(KeyError):
        propose(ctx, complete=bug, model="fake:m")
