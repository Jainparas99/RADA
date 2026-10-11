"""Static gate: checks on a harness patch before anything is run (SPEC §9).

Deterministic code only. Every check is made against the *result* of applying the patch as
well as its paths, so pointer tricks cannot move a change somewhere the path check missed.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from ratchet.config import load_yaml
from ratchet.contracts import PatchOp
from ratchet.gate import hashes
from ratchet.gate.leakage import LeakTerms, strings_in
from ratchet.harness.models import HarnessSpec
from ratchet.harness.patch import PatchError, apply_patch_raw, pointer_tokens

STATIC_CODES = [
    "PATCH_PATH_NOT_EDITABLE",
    "PROTECTED_FIELD_CHANGED",
    "MODEL_NOT_ALLOWED",
    "SCHEMA_INVALID",
    "OUT_OF_BOUNDS",
    "PATCH_TOO_LARGE",
    "HOLDOUT_LEAKAGE",
    "SCORER_TAMPERED",
    "GATE_TAMPERED",
]


class StaticReport(BaseModel):
    codes: list[str] = []
    details: dict[str, list[str]] = {}
    candidate: HarnessSpec | None = None  # set only when the patch applies and validates

    @property
    def ok(self) -> bool:
        return not self.codes

    def add(self, code: str, detail: str) -> None:
        self.details.setdefault(code, []).append(detail)
        self.codes = [c for c in STATIC_CODES if c in self.details]


def _added_chars(value: Any) -> int:
    if value is None:
        return 0
    return len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))


def _resolve(doc: Any, pointer: str) -> Any:
    for tok in pointer_tokens(pointer):
        if not isinstance(doc, dict) or tok not in doc:
            return None
        doc = doc[tok]
    return doc


def _check_bounds(doc: dict, bounds: dict, rep: StaticReport) -> None:
    for root, lim in (bounds.get("lists") or {}).items():
        items = doc.get(root)
        if not isinstance(items, list):
            continue
        if len(items) > lim["max_items"]:
            rep.add("OUT_OF_BOUNDS", f"{root}: {len(items)} items > {lim['max_items']}")
        for i, item in enumerate(items):
            if isinstance(item, str) and len(item) > lim["max_item_chars"]:
                rep.add(
                    "OUT_OF_BOUNDS", f"{root}[{i}]: {len(item)} chars > {lim['max_item_chars']}"
                )
    for pointer, (lo, hi) in (bounds.get("numeric") or {}).items():
        v = _resolve(doc, pointer)
        if isinstance(v, bool) or not isinstance(v, int | float):
            continue  # wrong type is SCHEMA_INVALID's job
        if not (lo <= v <= hi):  # also rejects NaN
            rep.add("OUT_OF_BOUNDS", f"{pointer}: {v} not in [{lo}, {hi}]")


def static_report(
    parent: HarnessSpec,
    patch: list[PatchOp],
    *,
    leak_terms: LeakTerms | None = None,
    bounds: dict | None = None,
    gate_cfg: dict | None = None,
    current_scorer_hash: str | None = None,
    current_gate_hash: str | None = None,
) -> StaticReport:
    bounds = bounds if bounds is not None else load_yaml("bounds.yaml")
    cfg = (gate_cfg if gate_cfg is not None else load_yaml("gate.yaml")).get("static", {})
    editable = set(bounds.get("editable_roots") or [])
    rep = StaticReport()

    # Tamper checks do not depend on the patch at all.
    if parent.protected.scorer_ref.source_hash != (current_scorer_hash or hashes.scorer_hash()):
        rep.add("SCORER_TAMPERED", "scorer source differs from the hash pinned by the parent")
    if parent.protected.gate_ref.source_hash != (current_gate_hash or hashes.gate_hash()):
        rep.add("GATE_TAMPERED", "gate source differs from the hash pinned by the parent")

    # Size.
    if len(patch) > cfg.get("max_ops", 4):
        rep.add("PATCH_TOO_LARGE", f"{len(patch)} ops > {cfg.get('max_ops', 4)}")
    added = sum(_added_chars(op.value) for op in patch if op.op != "remove")
    if added > cfg.get("max_added_chars", 1200):
        rep.add("PATCH_TOO_LARGE", f"{added} added chars > {cfg.get('max_added_chars', 1200)}")

    # Paths.
    for op in patch:
        try:
            tokens = pointer_tokens(op.path)
        except PatchError as e:
            rep.add("PATCH_PATH_NOT_EDITABLE", str(e))
            continue
        root = tokens[0] if tokens else ""
        if root == "protected":
            rep.add("PROTECTED_FIELD_CHANGED", f"{op.op} {op.path}")
        elif root not in editable:
            rep.add("PATCH_PATH_NOT_EDITABLE", f"{op.op} {op.path!r}")
        elif len(tokens) == 1 and op.op == "remove":
            rep.add("PATCH_PATH_NOT_EDITABLE", f"cannot remove top-level {op.path}")

    # Result of applying the patch.
    parent_doc = parent.to_doc()
    try:
        cand_doc = apply_patch_raw(parent_doc, patch)
    except PatchError as e:
        rep.add("SCHEMA_INVALID", str(e))
        return rep
    if not isinstance(cand_doc, dict):
        rep.add("SCHEMA_INVALID", "patched spec is not an object")
        return rep

    if cand_doc.get("protected") != parent_doc["protected"]:
        rep.add("PROTECTED_FIELD_CHANGED", "protected section differs after applying the patch")
    for key in set(cand_doc) | set(parent_doc):
        if key not in editable and key != "protected" and cand_doc.get(key) != parent_doc.get(key):
            rep.add("PATCH_PATH_NOT_EDITABLE", f"non-editable field {key!r} differs after patch")

    candidate: HarnessSpec | None = None
    try:
        candidate = HarnessSpec.from_doc(cand_doc)
    except ValidationError as e:
        for err in e.errors()[:5]:
            loc = "/".join(str(x) for x in err["loc"])
            rep.add("SCHEMA_INVALID", f"{loc}: {err['msg']}")

    _check_bounds(cand_doc, bounds, rep)

    routing = cand_doc.get("model_routing")
    allow = parent.protected.model_allowlist  # the parent's list, never the candidate's
    for role, model in routing.items() if isinstance(routing, dict) else []:
        if model not in allow:  # exact match: different casing is a different model id
            rep.add("MODEL_NOT_ALLOWED", f"model_routing.{role} = {model!r}")

    if leak_terms is not None:
        text = "\n".join(
            s for k, v in cand_doc.items() if k != "protected" for s in strings_in({k: v})
        )
        text += "\n" + "\n".join(op.path for op in patch)
        for term in leak_terms.find(text):
            rep.add("HOLDOUT_LEAKAGE", f"mentions held-out name {term!r}")

    if rep.ok:
        rep.candidate = candidate
    return rep


def check_static(parent: HarnessSpec, patch: list[PatchOp], **kwargs: Any) -> list[str]:
    """Reason codes for a patch against its parent; empty means the patch may be run."""
    return static_report(parent, patch, **kwargs).codes
