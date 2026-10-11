"""Applying a proposer patch (RFC 6902 subset) to a harness spec."""

from __future__ import annotations

import copy

import jsonpatch
from jsonpointer import JsonPointerException

from ratchet.contracts import PatchOp
from ratchet.harness.models import HarnessSpec


class PatchError(ValueError):
    """The patch is malformed or does not apply to the parent spec."""


def pointer_tokens(path: str) -> list[str]:
    """Unescaped JSON-pointer tokens. Raises PatchError unless `path` is "" or starts with "/"."""
    if path == "":
        return []
    if not path.startswith("/"):
        raise PatchError(f"path must start with '/': {path!r}")
    return [t.replace("~1", "/").replace("~0", "~") for t in path[1:].split("/")]


def patch_to_json(patch: list[PatchOp]) -> list[dict]:
    out: list[dict] = []
    for op in patch:
        d: dict = {"op": op.op, "path": op.path}
        if op.op != "remove":
            d["value"] = op.value
        out.append(d)
    return out


def apply_patch_raw(parent_doc: dict, patch: list[PatchOp]) -> dict:
    """Apply to a plain dict without validating the result. Never mutates `parent_doc`."""
    try:
        return jsonpatch.apply_patch(copy.deepcopy(parent_doc), patch_to_json(patch))
    except (jsonpatch.JsonPatchException, JsonPointerException, TypeError, KeyError) as e:
        raise PatchError(f"patch does not apply: {e}") from e


def apply_patch(parent: HarnessSpec, patch: list[PatchOp]) -> HarnessSpec:
    """Apply and validate. Call `gate.static.check_static` first: this does not police paths."""
    try:
        return HarnessSpec.from_doc(apply_patch_raw(parent.to_doc(), patch))
    except ValueError as e:
        raise PatchError(str(e)) from e
