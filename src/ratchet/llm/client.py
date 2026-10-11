"""`complete()`: the one entry point for every LLM call in Ratchet (CLAUDE.md invariant 9).

- Model ids are `provider:model`; providers without keys raise `ProviderUnavailable`.
- Every provider round-trip is cached in `llm_cache` under sha256(model, system, messages, params),
  so re-runs and `ratchet replay` are free and deterministic.
- Timeouts per request; exponential backoff on 408/409/429/5xx and connection errors.
- JSON mode (`json_schema=`): the reply is parsed and validated; on failure the validation error
  is appended and the model retried (up to `json_retries` times), then `LLMJSONError`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from ratchet.llm import cache as cache_mod
from ratchet.llm.jsonutil import JSONExtractError, extract_json, validate
from ratchet.llm.providers import Provider, build_provider, cost_usd, split_model
from ratchet.llm.types import (
    CacheMiss,
    LLMCallError,
    LLMJSONError,
    LLMResponse,
    ProviderUnavailable,
    RawCompletion,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT_S = 90.0
RETRYABLE_STATUS = {408, 409, 429}

Schema = dict | type[BaseModel]


def is_retryable(e: BaseException) -> bool:
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS or status >= 500
    if isinstance(e, TimeoutError | ConnectionError):
        return True
    names = " ".join(c.__name__ for c in type(e).__mro__)
    return "Timeout" in names or "Connection" in names


def _schema_dict(schema: Schema) -> dict:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    return schema


def json_instruction(schema: dict) -> str:
    return (
        "Reply with exactly one JSON value that validates against this JSON Schema. "
        "No prose, no markdown fences.\nJSON Schema:\n"
        + json.dumps(schema, separators=(",", ":"), sort_keys=True)
    )


class LLMClient:
    def __init__(
        self,
        cache: cache_mod.Cache | None = None,
        providers: dict[str, Provider] | None = None,
        provider_factory: Callable[[str], Provider] = build_provider,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = DEFAULT_TIMEOUT_S,
        json_retries: int = 2,
        cache_only: bool = False,
    ) -> None:
        self._cache = cache
        self._providers: dict[str, Provider] = dict(providers or {})
        self._factory = provider_factory
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.json_retries = json_retries
        self.cache_only = cache_only

    @property
    def cache(self) -> cache_mod.Cache:
        if self._cache is None:
            self._cache = cache_mod.default_cache()
        return self._cache

    # -- providers -----------------------------------------------------------------

    def provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = self._factory(name)  # may raise ProviderUnavailable
        return self._providers[name]

    def available(self, model: str) -> bool:
        """True if `model` can be called (valid id, key present, SDK installed)."""
        try:
            self.provider(split_model(model)[0])
            return True
        except Exception:
            return False

    # -- one round-trip ------------------------------------------------------------

    def raw(
        self,
        model: str,
        system: str | None,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> tuple[RawCompletion, bool]:
        provider_name, model_id = split_model(model)
        key = cache_mod.cache_key(
            "complete",
            model,
            {
                "system": system,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": json_mode,
            },
        )
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return cache_mod.raw_from_cache(hit), True
        if self.cache_only:
            raise CacheMiss(f"{model}: request not in llm_cache (cache-only mode)")
        prov = self.provider(provider_name)
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                out = prov.call(
                    model_id,
                    system,
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                    json_mode=json_mode,
                )
                break
            except ProviderUnavailable:
                raise
            except Exception as e:
                last = e
                if not is_retryable(e) or attempt == self.max_attempts:
                    raise LLMCallError(
                        f"{model}: {type(e).__name__} after {attempt} attempt(s)"
                    ) from e
                delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
                log.warning("%s: %s, retry %d in %.1fs", model, type(e).__name__, attempt, delay)
                self.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise LLMCallError(f"{model}: failed") from last
        if use_cache:
            self.cache.put(key, model, "complete", out.model_dump(mode="json"))
        return out, False

    # -- the public call -----------------------------------------------------------

    def complete(
        self,
        role: str = "",
        model: str = "",
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        json_schema: Schema | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_cache: bool = True,
    ) -> LLMResponse:
        if not model:
            raise ValueError("complete() needs model='provider:model'")
        # `{"role": "system"}` messages are folded into the system prompt.
        sys_parts = [system] if system else []
        convo: list[dict] = []
        for m in messages or []:
            if m.get("role") == "system":
                sys_parts.append(str(m.get("content", "")))
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        schema = _schema_dict(json_schema) if json_schema is not None else None
        if schema is not None:
            sys_parts.append(json_instruction(schema))
        sys_text = "\n\n".join(p for p in sys_parts if p) or None

        tin = tout = 0
        cost = 0.0
        all_cached = True
        errors: list[str] = []
        attempts = 1 + (self.json_retries if schema is not None else 0)
        for attempt in range(1, attempts + 1):
            out, cached = self.raw(
                model,
                sys_text,
                convo,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=schema is not None,
                use_cache=use_cache,
            )
            all_cached &= cached
            tin += out.input_tokens
            tout += out.output_tokens
            c = 0.0 if cached else cost_usd(model, out.input_tokens, out.output_tokens)
            cost += c
            log.info(
                "llm %s %s in=%d out=%d cost=$%.5f cached=%s",
                role or "-",
                model,
                out.input_tokens,
                out.output_tokens,
                c,
                cached,
            )
            resp = LLMResponse(
                text=out.text,
                role=role,
                model=model,
                input_tokens=tin,
                output_tokens=tout,
                cost_usd=cost,
                cached=all_cached,
                attempts=attempt,
                tool_calls=out.tool_calls,
                stop_reason=out.stop_reason,
            )
            if schema is None:
                try:
                    resp.data = json.loads(out.text)
                except ValueError:
                    pass
                return resp
            data, parsed, errors = _parse_and_validate(out.text, json_schema, schema)
            if not errors:
                resp.data, resp.parsed = data, parsed
                return resp
            if attempt < attempts:
                convo = [
                    *convo,
                    {"role": "assistant", "content": out.text or "(empty reply)"},
                    {
                        "role": "user",
                        "content": "Your reply was not valid:\n- "
                        + "\n- ".join(errors)
                        + "\nReply again with only the corrected JSON.",
                    },
                ]
        raise LLMJSONError(
            f"{model}: no schema-valid JSON after {attempts} attempts: {errors[:3]}",
            errors=errors,
            last_text=out.text,
            attempts=attempts,
        )


def _parse_and_validate(text: str, json_schema: Schema, schema: dict) -> tuple[Any, Any, list[str]]:
    try:
        data = extract_json(text)
    except JSONExtractError as e:
        return None, None, [str(e)]
    if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
        try:
            return data, json_schema.model_validate(data), []
        except ValidationError as e:
            return (
                data,
                None,
                [
                    f"$.{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in e.errors(include_url=False)[:8]
                ],
            )
    errs = validate(data, schema)
    return data, (data if not errs else None), errs


# -- module-level default client ---------------------------------------------------

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        flag = os.environ.get("RATCHET_LLM_CACHE_ONLY", "").strip().lower()
        _client = LLMClient(cache_only=flag in ("1", "true", "yes"))
    return _client


def set_client(client: LLMClient | None) -> None:
    """Swap the process-wide client (tests, replay's cache-only mode)."""
    global _client
    _client = client


def complete(
    role: str = "",
    model: str = "",
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    json_schema: Schema | None = None,
    **kwargs: Any,
) -> LLMResponse:
    return get_client().complete(role, model, messages, tools, json_schema, **kwargs)


def available(model: str) -> bool:
    return get_client().available(model)
