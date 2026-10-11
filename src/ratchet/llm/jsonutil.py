"""JSON extraction from model text and a small JSON-Schema validator.

The validator covers the subset pydantic emits (`$ref`/`$defs`, `anyOf`/`oneOf`/`allOf`, `type`,
`enum`, `const`, `properties`, `required`, `additionalProperties`, `items`, `prefixItems`,
`minItems`/`maxItems`). Unknown keywords are ignored. Errors are short, path-prefixed strings
meant to be fed back to the model in a repair turn.
"""

from __future__ import annotations

import json
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class JSONExtractError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Parse the JSON value in `text`, tolerating code fences and prose around one object."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start < 0:
            continue
        end = _matching(s, start, open_ch, close_ch)
        if end is not None:
            try:
                return json.loads(s[start : end + 1])
            except ValueError as e:
                raise JSONExtractError(f"reply contains malformed JSON: {e}") from e
    raise JSONExtractError("reply contains no JSON object")


def _matching(s: str, start: int, open_ch: str, close_ch: str) -> int | None:
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def validate(instance: Any, schema: dict, max_errors: int = 8) -> list[str]:
    errors: list[str] = []
    _check(instance, schema, schema, "$", errors)
    return errors[:max_errors]


def _resolve(schema: dict, root: dict) -> dict:
    seen = 0
    while "$ref" in schema and seen < 32:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return {}
        node: Any = root
        for part in ref[2:].split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        schema = {**node, **{k: v for k, v in schema.items() if k != "$ref"}}
        seen += 1
    return schema


def _type_ok(instance: Any, t: str) -> bool:
    if t in ("integer", "number") and isinstance(instance, bool):
        return False
    if t == "integer" and isinstance(instance, float):
        return instance.is_integer()
    return isinstance(instance, _TYPES.get(t, (object,)))


def _check(instance: Any, schema: Any, root: dict, path: str, errors: list[str]) -> None:
    if schema is False:
        errors.append(f"{path}: not allowed")
        return
    if not isinstance(schema, dict):
        return
    schema = _resolve(schema, root)

    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = []
            for sub in schema[key]:
                errs: list[str] = []
                _check(instance, sub, root, path, errs)
                if not errs:
                    break
                branches.append(errs)
            else:
                best = min(branches, key=len) if branches else []
                errors.append(f"{path}: matches none of {key} ({'; '.join(best[:2])})")
                return
    for sub in schema.get("allOf", []):
        _check(instance, sub, root, path, errors)

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(instance, x) for x in types):
            errors.append(f"{path}: expected {'/'.join(types)}, got {type(instance).__name__}")
            return

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required field {req!r}")
        extra = schema.get("additionalProperties", True)
        for k, v in instance.items():
            if k in props:
                _check(v, props[k], root, f"{path}.{k}", errors)
            elif extra is False:
                errors.append(f"{path}: unexpected field {k!r}")
            elif isinstance(extra, dict):
                _check(v, extra, root, f"{path}.{k}", errors)
    elif isinstance(instance, list | tuple):
        n = len(instance)
        if "minItems" in schema and n < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and n > schema["maxItems"]:
            errors.append(f"{path}: at most {schema['maxItems']} items")
        prefix = schema.get("prefixItems", [])
        for i, sub in enumerate(prefix[:n]):
            _check(instance[i], sub, root, f"{path}[{i}]", errors)
        items = schema.get("items")
        if isinstance(items, dict):
            for i in range(len(prefix), n):
                _check(instance[i], items, root, f"{path}[{i}]", errors)
