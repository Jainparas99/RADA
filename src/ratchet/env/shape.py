"""query_shape(): a stable, value-free description of a query (SPEC §8).

    movies | filter {genres: EQ, year: RANGE} | sort {imdb.rating: DESC, _id: ASC}
           | proj {imdb.rating, title} | limit
    comments | pipeline $lookup(from=movies, local=movie_id, foreign=_id, as=m)
             > $unwind(m) > $match {m.year: EQ} > $count

(wrapped here for width; the real string is one line)

Values are stripped, operators kept. Filter fields and projection fields are sorted (their
order has no meaning); sort keys and pipeline stages keep their order (it does).
"""

from __future__ import annotations

import re
from typing import Any

from bson.regex import Regex

from ratchet.contracts import QuerySpec

_RANGE = {"$gt", "$gte", "$lt", "$lte"}
_OP_TOKEN = {
    "$eq": "EQ",
    "$ne": "NE",
    "$in": "IN",
    "$nin": "NIN",
    "$exists": "EXISTS",
    "$size": "SIZE",
    "$all": "ALL",
    "$type": "TYPE",
    "$mod": "MOD",
    "$geoWithin": "GEO",
    "$geoIntersects": "GEO",
    "$near": "GEO",
    "$nearSphere": "GEO",
    "$bitsAllSet": "BITS",
    "$bitsAnySet": "BITS",
    "$bitsAllClear": "BITS",
    "$bitsAnyClear": "BITS",
}


def _regex_token(pattern: Any, options: str = "") -> str:
    if isinstance(pattern, (Regex, re.Pattern)):
        flags = pattern.flags
        options = options + ("i" if flags & re.IGNORECASE else "")
        pattern = pattern.pattern
    anchored = isinstance(pattern, str) and pattern.startswith("^") and "i" not in options
    return "REGEX_PREFIX" if anchored else "REGEX"


def _pred(v: Any) -> str:
    if isinstance(v, (Regex, re.Pattern)):
        return _regex_token(v)
    if isinstance(v, dict) and v and all(isinstance(k, str) and k.startswith("$") for k in v):
        toks: list[str] = []
        if _RANGE & set(v):
            toks.append("RANGE")
        for k, sub in v.items():
            if k in _RANGE or k == "$options":
                continue
            if k == "$regex":
                toks.append(_regex_token(sub, str(v.get("$options", ""))))
            elif k == "$elemMatch":
                toks.append("ELEM_MATCH" + (_filter(sub) if isinstance(sub, dict) else ""))
            elif k == "$not":
                toks.append(f"NOT({_pred(sub)})")
            else:
                toks.append(_OP_TOKEN.get(k, k.lstrip("$").upper()))
        return "+".join(sorted(dict.fromkeys(toks)))
    return "EQ"


def _collect(f: dict, out: dict[str, set[str]], extra: list[str]) -> None:
    for k, v in f.items():
        if k == "$and" and isinstance(v, list):
            for sub in v:
                if isinstance(sub, dict):
                    _collect(sub, out, extra)
        elif k in ("$or", "$nor") and isinstance(v, list):
            branches = sorted(_filter(sub) for sub in v if isinstance(sub, dict))
            extra.append(f"{k[1:].upper()}[{', '.join(branches)}]")
        elif k == "$text":
            extra.append("$text: TEXT")
        elif k == "$expr":
            extra.append("$expr: EXPR")
        elif k == "$where":
            extra.append("$where: JS")
        elif k.startswith("$"):
            extra.append(f"{k}: {k[1:].upper()}")
        else:
            out.setdefault(k, set()).add(_pred(v))


def _filter(f: dict) -> str:
    fields: dict[str, set[str]] = {}
    extra: list[str] = []
    _collect(f or {}, fields, extra)
    parts = [f"{k}: {'+'.join(sorted(v))}" for k, v in sorted(fields.items())]
    parts.extend(sorted(extra))
    return "{" + ", ".join(parts) + "}"


def _sort(pairs: list[tuple[str, Any]]) -> str:
    def d(v: Any) -> str:
        if isinstance(v, dict):
            return "META"
        return "ASC" if v == 1 else "DESC" if v == -1 else str(v).upper()

    return "{" + ", ".join(f"{k}: {d(v)}" for k, v in pairs) + "}"


def _proj(p: dict) -> str:
    names = []
    for k, v in p.items():
        if v in (0, False):
            names.append(f"-{k}")
        elif isinstance(v, dict) or (isinstance(v, str) and v.startswith("$")):
            names.append(f"{k}=EXPR")
        else:
            names.append(k)
    return "{" + ", ".join(sorted(names)) + "}"


def _field(v: Any) -> str:
    return v.lstrip("$") if isinstance(v, str) else "EXPR"


def _stage(st: dict) -> str:
    if not isinstance(st, dict) or len(st) != 1:
        return "?"
    name, spec = next(iter(st.items()))
    if name == "$match":
        return f"$match {_filter(spec)}"
    if name == "$sort":
        return f"$sort {_sort(list(spec.items()))}"
    if name in ("$project", "$addFields", "$set"):
        return f"{name} {_proj(spec)}" if isinstance(spec, dict) else name
    if name == "$unset":
        return f"$unset {spec if isinstance(spec, str) else sorted(spec)}"
    if name == "$unwind":
        path = spec if isinstance(spec, str) else spec.get("path", "")
        return f"$unwind({_field(path)})"
    if name == "$lookup":
        if "localField" in spec:
            inner = (
                f"from={spec.get('from')}, local={spec.get('localField')}, "
                f"foreign={spec.get('foreignField')}"
            )
        else:
            inner = f"from={spec.get('from')}, let={sorted((spec.get('let') or {}).keys())}"
        if "pipeline" in spec:
            inner += ", pipeline=[" + " > ".join(_stage(s) for s in spec["pipeline"]) + "]"
        return f"$lookup({inner}, as={spec.get('as')})"
    if name == "$group":
        gid = spec.get("_id")
        if isinstance(gid, dict):
            by = "{" + ", ".join(sorted(gid)) + "}"
        elif gid is None:
            by = "null"
        else:
            by = _field(gid)
        accs = sorted(
            {
                next(iter(v)).lstrip("$")
                for k, v in spec.items()
                if k != "_id" and isinstance(v, dict) and v
            }
        )
        return f"$group(by={by}, acc=[{', '.join(accs)}])"
    if name in ("$limit", "$skip", "$count", "$sortByCount"):
        return name
    if name == "$facet":
        return (
            "$facet{"
            + ", ".join(
                f"{k}: [" + " > ".join(_stage(s) for s in v) + "]" for k, v in sorted(spec.items())
            )
            + "}"
        )
    if name == "$unionWith":
        coll = spec if isinstance(spec, str) else spec.get("coll")
        return f"$unionWith({coll})"
    return name


def query_shape(q: QuerySpec) -> str:
    parts = [q.collection]
    if q.kind == "find":
        parts.append(f"filter {_filter(q.filter)}")
        if q.sort:
            parts.append(f"sort {_sort(list(q.sort))}")
        if q.projection:
            parts.append(f"proj {_proj(q.projection)}")
        if q.limit is not None:
            parts.append("limit")
    else:
        parts.append("pipeline " + " > ".join(_stage(s) for s in (q.pipeline or [])))
    return " | ".join(parts)
