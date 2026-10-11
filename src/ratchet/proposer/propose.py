"""The proposer: one LLM call that turns train traces into a JSON patch over the harness."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from ratchet.config import get_settings
from ratchet.contracts import ProposerOutput
from ratchet.proposer.context import ProposerContext

ROLE = "proposer"

SYSTEM = """You improve a harness that steers a MongoDB query-tuning agent.

The harness is a JSON document: planner rules, a critic rubric, context/retry/critic/memory
policies and model routing. You are given the current harness, how the agent did on TRAINING
workloads (with raw traces of the worst runs), and why recent patches were rejected.

Propose ONE small change that should make the agent tune *unseen* workloads better. A separate
deterministic gate decides: it promotes a patch only if held-out workloads improve and no query
result changes. You never see those workloads, so:
- write general principles (index key order, when to move a $match, when not to add an index),
  never anything about a specific collection, field or workload id;
- do not repeat a patch that was already rejected for the same reason;
- prefer fewer tokens when results would be equal.

Rules for the patch (RFC 6902, ops add/replace/remove only):
- paths must be under the editable paths you are given; never touch /protected;
- stay inside the given bounds and patch limits;
- model_routing values must come from protected.model_allowlist.

Reply with one JSON object and nothing else:
{"diagnosis": str, "hypothesis": str, "patch": [{"op": ..., "path": ..., "value": ...}],
 "expected_effect": str}"""

CompleteFn = Callable[..., Any]


class ProposerError(RuntimeError):
    """The proposer did not return a valid ProposerOutput, even after one repair attempt."""


def build_messages(context: ProposerContext) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": context.model_dump_json(exclude={"redactions"})},
    ]


def _default_complete() -> CompleteFn:
    from ratchet.llm import complete  # Track C  # type: ignore[attr-defined]

    return complete


def response_text(resp: Any) -> str:
    for attr in ("text", "content"):
        v = getattr(resp, attr, None)
        if isinstance(v, str):
            return v
    return resp if isinstance(resp, str) else json.dumps(getattr(resp, "data", None))


def response_tokens(resp: Any) -> int:
    v = getattr(resp, "tokens", None)
    if isinstance(v, int):
        return v
    return int(getattr(resp, "input_tokens", 0) or 0) + int(getattr(resp, "output_tokens", 0) or 0)


def parse_output(resp: Any) -> ProposerOutput:
    """Strict ProposerOutput from an LLM response; raises ValueError with a repairable message."""
    data = getattr(resp, "data", None)
    if not isinstance(data, dict):
        text = response_text(resp).strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"reply is not valid JSON: {e}") from e
    try:
        return ProposerOutput.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"reply does not match the schema: {e.errors(include_url=False)}") from e


def _is_llm_error(e: Exception) -> bool:
    """True for Track C's `ratchet.llm.LLMError` family (matched by name: no hard import)."""
    return any(c.__name__ == "LLMError" for c in type(e).__mro__)


def propose(
    context: ProposerContext,
    complete: CompleteFn | None = None,
    model: str | None = None,
    usage: dict | None = None,
) -> ProposerOutput:
    """Ask the proposer for a patch. One repair retry on invalid JSON, then ProposerError."""
    complete = complete or _default_complete()
    model = model or get_settings().proposer_model
    schema = ProposerOutput.model_json_schema()
    messages = build_messages(context)
    last_error = ""
    for _attempt in range(2):
        try:
            resp = complete(ROLE, model, messages, json_schema=schema)
        except Exception as e:
            # ratchet.llm validates and repairs JSON itself, then raises; so do provider outages.
            if _is_llm_error(e):
                raise ProposerError(f"{type(e).__name__}: {e}") from e
            raise
        if usage is not None:
            usage["tokens"] = usage.get("tokens", 0) + response_tokens(resp)
            usage["calls"] = usage.get("calls", 0) + 1
            usage["model"] = model
        try:
            return parse_output(resp)
        except ValueError as e:
            last_error = str(e)
            messages = [
                *messages,
                {"role": "assistant", "content": response_text(resp)},
                {
                    "role": "user",
                    "content": f"That was not usable: {last_error[:800]}\n"
                    "Reply again with only the JSON object in the required shape.",
                },
            ]
    raise ProposerError(last_error)
