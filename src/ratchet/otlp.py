"""Export stored tuning traces as OpenTelemetry (OTLP/HTTP JSON) spans.

Raindrop Workshop, a local agent debugger, accepts OTLP JSON at `/v1/traces` (default port
5899). One trace per tuning run: a root span for the run, child spans for each step, and LLM
calls tagged with the OpenTelemetry GenAI attribute names so they render as model calls.

Stored traces carry durations for LLM calls only, so span times are laid out sequentially from
the export time; ordering and LLM latency are real, absolute timestamps are not.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from typing import Any

from ratchet.contracts import TuneResult
from ratchet.jsonsafe import jsonable

DEFAULT_ENDPOINT = "http://localhost:5899/v1/traces"
MAX_ATTR_CHARS = 8000
_STEP_NS = 1_000_000  # 1 ms for steps that carry no duration


def _hex(seed: str, n: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def _attr(key: str, value: Any) -> dict:
    if isinstance(value, bool):
        v: dict = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        text = value if isinstance(value, str) else json.dumps(jsonable(value), default=str)
        v = {"stringValue": text[:MAX_ATTR_CHARS]}
    return {"key": key, "value": v}


def _attrs(pairs: dict[str, Any]) -> list[dict]:
    return [_attr(k, v) for k, v in pairs.items() if v is not None]


def _step_span(step: dict, i: int, result: TuneResult) -> tuple[str, dict[str, Any], int]:
    """(name, attributes, duration in ns) for one trace entry."""
    kind = str(step.get("step", "step"))
    if kind == "llm":
        provider, _, model = str(step.get("model", "")).partition(":")
        messages = step.get("messages") or []
        reply = step.get("response")
        reply = reply if isinstance(reply, str) or reply is None else json.dumps(jsonable(reply))
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": provider,
            "gen_ai.request.model": model or provider,
            "gen_ai.usage.total_tokens": step.get("tokens"),
            # The same content under each naming scheme Raindrop Workshop reads: OpenTelemetry
            # GenAI, the Vercel AI SDK and OpenLLMetry (indexed prompt/completion).
            "gen_ai.input.messages": messages,
            "gen_ai.output.messages": [{"role": "assistant", "content": reply}] if reply else None,
            "ai.prompt.messages": messages,
            "ai.response.text": reply,
            "ai.model.id": model or provider,
            "ai.model.provider": provider,
            "gen_ai.completion.0.role": "assistant" if reply else None,
            "gen_ai.completion.0.content": reply,
            "traceloop.span.kind": "llm",
            "llm.request.type": "chat",
            "ratchet.role": step.get("role"),
            "ratchet.cached": step.get("cached"),
            "ratchet.error": step.get("error"),
        }
        for n, m in enumerate(messages if isinstance(messages, list) else []):
            if isinstance(m, dict):
                attrs[f"gen_ai.prompt.{n}.role"] = m.get("role")
                attrs[f"gen_ai.prompt.{n}.content"] = m.get("content")
        duration = int(step.get("ms") or 1) * 1_000_000
        return f"chat {model or provider} ({step.get('role', 'llm')})", attrs, duration
    attrs = {f"ratchet.{k}": v for k, v in step.items() if k != "step"}
    attrs["traceloop.span.kind"] = "task"
    attrs["traceloop.entity.name"] = kind
    attrs["traceloop.entity.output"] = {k: v for k, v in step.items() if k != "step"}
    if kind == "apply_measure" and isinstance(step.get("score"), dict):
        attrs["ratchet.score.value"] = step["score"].get("score")
        attrs["ratchet.score.violations"] = step["score"].get("violations")
    return kind, attrs, _STEP_NS


def result_spans(result: TuneResult, start_ns: int) -> list[dict]:
    """OTLP spans for one tuning run, root first."""
    seed = f"{result.harness_version}/{result.workload_id}"
    trace_id, root_id = _hex(seed, 32), _hex(seed + "/root", 16)
    spans: list[dict] = []
    cursor = start_ns
    for i, step in enumerate(result.trace):
        name, attrs, duration = _step_span(step, i, result)
        failed = bool(step.get("error"))
        spans.append(
            {
                "traceId": trace_id,
                "spanId": _hex(f"{seed}/{i}", 16),
                "parentSpanId": root_id,
                "name": name,
                "kind": 3 if step.get("step") == "llm" else 1,  # CLIENT for model calls
                "startTimeUnixNano": str(cursor),
                "endTimeUnixNano": str(cursor + duration),
                "attributes": _attrs(attrs),
                "status": {"code": 2 if failed else 1},
            }
        )
        cursor += duration
    violations = result.score.violations
    root = {
        "traceId": trace_id,
        "spanId": root_id,
        "name": f"tune {result.workload_id}",
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(max(cursor, start_ns + _STEP_NS)),
        "attributes": _attrs(
            {
                "traceloop.span.kind": "workflow",
                "traceloop.entity.name": f"tune {result.workload_id}",
                "traceloop.association.properties.event_name": "tune",
                "traceloop.entity.input": {
                    "workload": result.workload_id,
                    "split": result.split,
                    "harness_version": result.harness_version,
                },
                "traceloop.entity.output": {
                    "score": result.score.model_dump(),
                    "plan": result.plan.summary,
                    "cost_before": result.before.workload_cost,
                    "cost_after": result.after.workload_cost,
                },
                "ratchet.workload": result.workload_id,
                "ratchet.split": result.split,
                "ratchet.harness_version": result.harness_version,
                "ratchet.score": result.score.score,
                "ratchet.improvement": result.score.improvement,
                "ratchet.index_penalty": result.score.index_penalty,
                "ratchet.violations": ",".join(violations),
                "ratchet.cost_before": result.before.workload_cost,
                "ratchet.cost_after": result.after.workload_cost,
                "ratchet.tokens": result.tokens,
                "ratchet.rounds": result.rounds,
                "ratchet.plan": result.plan.summary,
                "ratchet.self_diagnostics": result.self_diagnostics or None,
                "ratchet.lessons_used": result.lessons_used or None,
            }
        ),
        "status": {"code": 2 if violations else 1, "message": ",".join(violations)},
    }
    return [root, *spans]


def to_otlp(results: list[TuneResult], now_ns: int | None = None) -> dict:
    """An OTLP ExportTraceServiceRequest (JSON encoding) for a list of tuning runs."""
    now_ns = now_ns if now_ns is not None else time.time_ns()
    spans: list[dict] = []
    for i, r in enumerate(results):
        spans += result_spans(r, now_ns + i * 1_000_000_000)
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs({"service.name": "ratchet"})},
                "scopeSpans": [{"scope": {"name": "ratchet.tune"}, "spans": spans}],
            }
        ]
    }


def post(payload: dict, endpoint: str = DEFAULT_ENDPOINT, timeout: float = 10.0) -> int:
    """POST an OTLP JSON payload; returns the HTTP status. Raises OSError if unreachable."""
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local debugger)
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
