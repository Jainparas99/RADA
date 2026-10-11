"""Every static reason code has a test that triggers it and one that doesn't (CLAUDE.md)."""

import pytest

from ratchet.contracts import PatchOp, Workload
from ratchet.gate import hashes
from ratchet.gate.leakage import LeakTerms, normalise
from ratchet.gate.static import STATIC_CODES, check_static, static_report
from tests.fakes import fake_workloads


def P(op, path, value=None):  # noqa: N802
    return PatchOp(op=op, path=path, value=value)


GOOD = [
    [P("add", "/planner_rules/-", "Order compound index keys: equality, then sort, then range.")],
    [P("replace", "/context_policy/include_detector_findings", True)],
    [P("replace", "/retry_policy/max_rounds", 2), P("replace", "/critic_policy/enabled", True)],
    [P("replace", "/memory_policy/inject/mode", "on_retry")],
    [P("replace", "/memory_policy/retrieve", {"k": 5, "min_similarity": 0.8})],
    [P("remove", "/planner_rules/0")],
    [P("replace", "/model_routing/planner", "anthropic:claude-sonnet-5-5")],
    [],
]


@pytest.mark.parametrize("patch", GOOD)
def test_good_patches_pass_every_check(spec, leak_terms, patch):
    rep = static_report(spec, patch, leak_terms=leak_terms)
    assert rep.codes == [] and rep.candidate is not None


def codes(spec, patch, **kw):
    return check_static(spec, patch, **kw)


# ---------------------------------------------------------------- PATCH_PATH_NOT_EDITABLE


@pytest.mark.parametrize(
    "patch",
    [
        [P("add", "/scores", {"holdout": 1.0})],
        [P("add", "/status", "accepted")],
        [P("replace", "", {})],
        [P("replace", "planner_rules/0", "no leading slash")],
        [P("add", "/model_routing~1planner", "x")],
        [P("remove", "/planner_rules")],
        [P("add", "/Planner_Rules", ["case trick"])],
    ],
)
def test_patch_path_not_editable(spec, patch):
    assert "PATCH_PATH_NOT_EDITABLE" in codes(spec, patch)


def test_editable_paths_do_not_fire_not_editable(spec):
    assert "PATCH_PATH_NOT_EDITABLE" not in codes(spec, [P("replace", "/critic_rubric", ["a"])])


# ---------------------------------------------------------------- PROTECTED_FIELD_CHANGED


@pytest.mark.parametrize(
    "patch",
    [
        [P("replace", "/protected/token_budget_per_workload", 10**9)],
        [P("add", "/protected/model_allowlist/-", "openai:whatever")],
        [P("replace", "/protected/allowed_actions", ["create_index", "$out"])],
        [P("remove", "/protected/gate_ref")],
        [P("add", "/protected/new_key", 1)],
        [P("replace", "/protected", {})],
        [P("add", "/protected/../planner_rules/-", "dot-dot trick")],
        [P("replace", "/protected/scorer_ref/source_hash", "sha256:mine")],
        # a no-op write under protected is still a write under protected
        [P("replace", "/protected/token_budget_per_workload", 30000)],
    ],
)
def test_protected_field_changed(spec, patch):
    assert "PROTECTED_FIELD_CHANGED" in codes(spec, patch)


def test_allowlist_edit_cannot_unlock_a_model(spec):
    patch = [
        P("add", "/protected/model_allowlist/-", "evil:model"),
        P("replace", "/model_routing/planner", "evil:model"),
    ]
    got = codes(spec, patch)
    assert "PROTECTED_FIELD_CHANGED" in got and "MODEL_NOT_ALLOWED" in got


def test_whole_document_replace_cannot_change_protected(spec):
    doc = spec.to_doc()
    doc["protected"]["token_budget_per_workload"] = 10**9
    got = codes(spec, [P("replace", "", doc)])
    assert "PROTECTED_FIELD_CHANGED" in got and "PATCH_PATH_NOT_EDITABLE" in got


# ---------------------------------------------------------------- MODEL_NOT_ALLOWED


@pytest.mark.parametrize(
    "model",
    ["openai:gpt-not-listed", "ANTHROPIC:CLAUDE-HAIKU-5-5", "anthropic:claude-haiku-5-5 ", ""],
)
def test_model_not_allowed(spec, model):
    assert "MODEL_NOT_ALLOWED" in codes(spec, [P("replace", "/model_routing/planner", model)])


def test_allowlisted_model_is_allowed(spec):
    for m in spec.protected.model_allowlist:
        assert codes(spec, [P("replace", "/model_routing/critic", m)]) == []


# ---------------------------------------------------------------- SCHEMA_INVALID


@pytest.mark.parametrize(
    "patch",
    [
        [P("replace", "/retry_policy/max_rounds", "two")],
        [P("replace", "/critic_policy/enabled", 1)],
        [P("replace", "/memory_policy/inject/mode", "sometimes")],
        [P("add", "/context_policy/secret_knob", True)],
        [P("add", "/planner_rules/-", {"not": "a string"})],
        [P("remove", "/retry_policy/max_rounds")],
        [P("remove", "/planner_rules/40")],
        [P("replace", "/planner_rules/../protected", {})],
        [P("add", "/model_routing/proposer", "anthropic:claude-sonnet-5-5")],
        [P("replace", "/memory_policy/write/from", ["holdout_improved"])],
    ],
)
def test_schema_invalid(spec, patch):
    assert "SCHEMA_INVALID" in codes(spec, patch)


def test_valid_types_do_not_fire_schema_invalid(spec):
    assert "SCHEMA_INVALID" not in codes(spec, [P("replace", "/retry_policy/max_rounds", 99)])


# ---------------------------------------------------------------- OUT_OF_BOUNDS


@pytest.mark.parametrize(
    "patch",
    [
        [P("replace", "/retry_policy/max_rounds", 99)],
        [P("replace", "/retry_policy/max_rounds", 0)],
        [P("replace", "/memory_policy/retrieve/k", 9)],
        [P("replace", "/memory_policy/retrieve/min_similarity", 0.1)],
        [P("replace", "/memory_policy/inject/max_tokens", 1501)],
        [P("replace", "/memory_policy/consolidate/merge_similarity", 0.99)],
        [P("replace", "/memory_policy/invalidate/after_harmful_uses", 0)],
        [P("replace", "/context_policy/max_chars", 10**7)],
        [P("add", "/planner_rules/-", "x" * 301)],
        [P("replace", "/critic_rubric", [f"rule {i}" for i in range(16)])],
    ],
)
def test_out_of_bounds(spec, patch):
    assert "OUT_OF_BOUNDS" in codes(spec, patch)


@pytest.mark.parametrize(
    "patch",
    [
        [P("replace", "/memory_policy/retrieve/k", 8)],
        [P("replace", "/memory_policy/retrieve/k", 0)],
        [P("replace", "/memory_policy/retrieve/min_similarity", 0.95)],
        [P("add", "/planner_rules/-", "x" * 300)],
        [P("replace", "/critic_rubric", [f"rule {i}" for i in range(15)])],
    ],
)
def test_values_on_the_boundary_are_in_bounds(spec, patch):
    assert codes(spec, patch) == []


# ---------------------------------------------------------------- PATCH_TOO_LARGE


def test_patch_too_large_by_ops(spec):
    five = [P("add", "/planner_rules/-", f"rule {i}") for i in range(5)]
    assert codes(spec, five) == ["PATCH_TOO_LARGE"]
    assert codes(spec, five[:4]) == []


def test_patch_too_large_by_added_chars(spec):
    big = [P("add", "/planner_rules/-", "y" * 300) for _ in range(4)]
    assert codes(spec, big) == []  # exactly 1200
    assert codes(spec, [*big[:3], P("add", "/planner_rules/-", "y" * 300 + "!")]) == [
        "OUT_OF_BOUNDS",
        "PATCH_TOO_LARGE",
    ]
    nested = [P("replace", "/critic_rubric", ["z" * 290] * 5)]
    assert codes(spec, nested) == ["PATCH_TOO_LARGE"]


# ---------------------------------------------------------------- HOLDOUT_LEAKAGE


def test_leak_terms_are_holdout_and_canary_only(leak_terms):
    for hidden in ("sales", "sold_at", "accounts", "fake_holdout_01", "trips", "line_items"):
        assert hidden in leak_terms.terms
    for seen in ("movies", "genres", "rating", "title", "fake_train_01", "q1", "fake_db", "_id"):
        assert seen not in leak_terms.terms


@pytest.mark.parametrize(
    "rule",
    [
        "For collection sales always index sold_at.",
        "For collection SALES always index it.",
        "Index `accounts.balance` first.",
        "Remember fake_holdout_01 needs two indexes.",
        "Index s​ales by store.",  # zero-width space
        "Index ѕаlеѕ by store.",  # Cyrillic look-alikes
        "Index ｓａｌｅｓ by store.",  # full-width
        "Always index sold-at / s o l d _ a t on that collection.",
        "trip_id lookups need an index.",
        "Prefer covering indexes.\nsales: index store first.",
    ],
)
def test_holdout_leakage_in_rules(spec, leak_terms, rule):
    patch = [P("add", "/planner_rules/-", rule)]
    assert codes(spec, patch, leak_terms=leak_terms) == ["HOLDOUT_LEAKAGE"]


def test_holdout_leakage_anywhere_in_the_editable_spec(spec, leak_terms):
    assert "HOLDOUT_LEAKAGE" in codes(
        spec,
        [P("replace", "/critic_rubric", ["ok", "check line_items joins"])],
        leak_terms=leak_terms,
    )
    assert "HOLDOUT_LEAKAGE" in codes(
        spec, [P("add", "/context_policy/sold_at", True)], leak_terms=leak_terms
    )


@pytest.mark.parametrize(
    "rule",
    [
        "Order compound index keys: equality, then sort, then range.",
        "For movies, index genres before year.",  # train names are fine
        "Wholesale changes are risky; prefer one index per query shape.",  # 'sale' inside a word
        "Move $match before $lookup when its fields exist on the source collection.",
    ],
)
def test_general_or_train_only_rules_do_not_leak(spec, leak_terms, rule):
    assert codes(spec, [P("add", "/planner_rules/-", rule)], leak_terms=leak_terms) == []


def test_ignore_terms_and_min_length():
    wl = fake_workloads()
    assert "sales" not in LeakTerms.from_workloads(wl, ignore_terms=["Sales"]).terms
    odd = Workload.model_validate(
        {**wl[3].model_dump(mode="json"), "id": "hx", "collections": ["ab", "sales"]}
    )
    assert "ab" not in LeakTerms.from_workloads([*wl, odd]).terms


def test_normalise():
    assert normalise("ＳＡlеs​") == "sales"


# ---------------------------------------------------------------- SCORER_TAMPERED / GATE_TAMPERED


def test_untampered_sources_pass(spec):
    assert codes(spec, []) == []


def test_scorer_tampered(spec):
    assert codes(spec, [], current_scorer_hash="sha256:edited") == ["SCORER_TAMPERED"]


def test_gate_tampered(spec):
    assert codes(spec, [], current_gate_hash="sha256:edited") == ["GATE_TAMPERED"]


def test_hashes_change_when_a_source_file_changes(tmp_path, monkeypatch):
    for name in ("gate.yaml", "bounds.yaml", "scoring.yaml"):
        (tmp_path / name).write_text("a: 1\n")
    monkeypatch.setenv("RATCHET_CONFIG_DIR", str(tmp_path))
    gate_before, scorer_before = hashes.gate_hash(), hashes.scorer_hash()
    (tmp_path / "gate.yaml").write_text("a: 2\n")
    assert hashes.gate_hash() != gate_before and hashes.scorer_hash() == scorer_before
    (tmp_path / "scoring.yaml").write_text("a: 2\n")
    assert hashes.scorer_hash() != scorer_before


def test_a_spec_pinned_to_old_source_blocks_every_patch(spec, tmp_path, monkeypatch):
    for name in ("gate.yaml", "bounds.yaml"):
        (tmp_path / name).write_text(open(f"config/{name}").read())
    (tmp_path / "gate.yaml").write_text(open("config/gate.yaml").read() + "\n# loosened\n")
    monkeypatch.setenv("RATCHET_CONFIG_DIR", str(tmp_path))
    assert "GATE_TAMPERED" in codes(spec, [P("add", "/planner_rules/-", "fine rule")])


# ---------------------------------------------------------------- reporting


def test_codes_are_reported_in_canonical_order_with_details(spec, leak_terms):
    rep = static_report(
        spec,
        [
            P("replace", "/model_routing/planner", "nope:model"),
            P("replace", "/protected/token_budget_per_workload", 1),
            P("add", "/planner_rules/-", "index sales"),
        ],
        leak_terms=leak_terms,
    )
    assert rep.codes == ["PROTECTED_FIELD_CHANGED", "MODEL_NOT_ALLOWED", "HOLDOUT_LEAKAGE"]
    assert rep.candidate is None and set(rep.details) == set(rep.codes)
    assert set(rep.codes) <= set(STATIC_CODES)


def test_parent_is_never_mutated(spec):
    before = spec.to_doc()
    check_static(spec, [P("add", "/planner_rules/-", "x"), P("replace", "/protected", {})])
    assert spec.to_doc() == before
