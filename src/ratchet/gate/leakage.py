"""HOLDOUT_LEAKAGE: names that only exist in holdout/canary workloads must not appear in
harness text or lessons. Deterministic string matching; only the gate ever reads these terms.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel

from ratchet.config import load_yaml
from ratchet.contracts import QuerySpec, Workload

# Common look-alikes for ASCII letters (Cyrillic / Greek), applied after NFKC + casefold.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
        "ѕ": "s", "ј": "j", "ԁ": "d", "һ": "h", "к": "k", "м": "m", "т": "t", "в": "b",
        "н": "h", "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "ν": "v",
        "τ": "t", "υ": "u", "χ": "x", "β": "b",
    }
)  # fmt: skip
_TOKEN = r"a-z0-9"


def normalise(text: str) -> str:
    """NFKC, casefold, drop zero-width/combining/control characters, fold look-alikes."""
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text))
    text = re.sub(r"\s", " ", text)  # keep word boundaries before control chars are dropped
    text = "".join(c for c in text if unicodedata.category(c) not in {"Mn", "Cf", "Cc", "Me"})
    return text.casefold().translate(_HOMOGLYPHS)


def _squash(text: str) -> str:
    return re.sub(rf"[^{_TOKEN}]", "", text)


def _names_in(obj: Any, out: set[str]) -> None:
    """Field names in a filter/pipeline: non-operator keys and "$field" references."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and not k.startswith("$"):
                out.add(k)
            if k in ("localField", "foreignField", "as", "from", "path") and isinstance(v, str):
                out.add(v.lstrip("$"))
            _names_in(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _names_in(v, out)
    elif isinstance(obj, str) and obj.startswith("$") and not obj.startswith("$$"):
        out.add(obj[1:])


def query_names(q: QuerySpec) -> set[str]:
    out: set[str] = {q.collection}
    _names_in(q.filter, out)
    _names_in(q.projection or {}, out)
    _names_in(q.pipeline or [], out)
    out.update(f for f, _ in q.sort or [])
    return out


def workload_names(w: Workload) -> set[str]:
    """Every identifier a workload exposes: ids, db, collections, fields, index names."""
    out: set[str] = {w.id, w.db, *w.collections}
    for q in w.queries:
        out |= query_names(q)
        out.add(f"{w.id}.{q.id}")
    for ixs in w.baseline_indexes.values():
        for ix in ixs:
            out.update(k for k, _ in ix.keys)
            if ix.name:
                out.add(ix.name)
    # dotted paths also leak through their components
    for name in list(out):
        out.update(p for p in name.split(".") if p)
    return {normalise(n) for n in out if n}


class LeakTerms(BaseModel):
    """Normalised names unique to holdout/canary workloads."""

    terms: list[str] = []
    squash_min_chars: int = 6

    @classmethod
    def from_workloads(
        cls,
        workloads: list[Workload],
        min_term_chars: int | None = None,
        squash_min_chars: int | None = None,
        ignore_terms: list[str] | None = None,
    ) -> LeakTerms:
        """Anything left as None comes from `config/gate.yaml` (static.leakage), so every
        caller (evolve, attack, tests) applies the same matching rules."""
        cfg = load_yaml("gate.yaml").get("static", {}).get("leakage", {})
        if min_term_chars is None:
            min_term_chars = cfg.get("min_term_chars", 3)
        if squash_min_chars is None:
            squash_min_chars = cfg.get("squash_min_chars", 6)
        if ignore_terms is None:
            ignore_terms = cfg.get("ignore_terms", [])
        seen: set[str] = set()
        hidden: set[str] = set()
        for w in workloads:
            (seen if w.split == "train" else hidden).update(workload_names(w))
        ignore = {normalise(t) for t in ignore_terms or []}
        terms = {
            t
            for t in hidden - seen - ignore
            if len(_squash(t)) >= min_term_chars and not t.isdigit() and t != "_id"
        }
        return cls(terms=sorted(terms), squash_min_chars=squash_min_chars)

    def find(self, text: str) -> list[str]:
        """Leak terms present in `text` (whole-token match; long terms also match squashed)."""
        norm = normalise(text)
        squashed = _squash(norm)
        hits: list[str] = []
        for t in self.terms:
            if re.search(rf"(?<![{_TOKEN}]){re.escape(t)}(?![{_TOKEN}])", norm):
                hits.append(t)
            elif len(_squash(t)) >= self.squash_min_chars and _squash(t) in squashed:
                hits.append(t)
        return hits


def strings_in(obj: Any) -> list[str]:
    """All strings (keys and values) in a JSON-like object."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in (*strings_in(k), *strings_in(v))]
    if isinstance(obj, list | tuple):
        return [s for v in obj for s in strings_in(v)]
    return []
