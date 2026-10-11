"""Lesson memory (SPEC §8): write from train runs, retrieve by query shape, consolidate,
account for helpful/harmful uses, invalidate.

Invariants enforced here (CLAUDE.md 7): lessons are written only from TRAIN results, and every
write (including counters) is refused while the store is frozen for holdout/canary evaluation.
Lesson text that mentions a held-out name is never stored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pymongo.database import Database

from ratchet.contracts import QuerySpec, TuneResult
from ratchet.db import MEMORIES
from ratchet.gate.leakage import LeakTerms
from ratchet.jsonsafe import jsonable
from ratchet.memory.models import Lesson, LessonSource

VECTOR_INDEX = "vector_index"
ROLE = "memory"
_EPS = 1e-9

SYSTEM = """You distil ONE reusable lesson from a MongoDB query-tuning attempt.

You get the query shapes, what the detectors found, the plan that was applied and how it scored.
Write a single general principle that would help tune a DIFFERENT workload with a similar query
shape: what to do, or (if the attempt broke a rule) what never to do.

- At most {max_chars} characters, one or two sentences, imperative.
- Never name a collection, field, index, workload or query id. Talk about roles instead:
  "the equality field", "the sort field", "the foreign key of the $lookup".
- Pick the query the lesson is about.

Reply with one JSON object and nothing else: {{"query_id": str, "lesson": str}}"""


class MemoryFrozenError(RuntimeError):
    """A write was attempted while memory is frozen (holdout/canary evaluation)."""


EmbedFn = Callable[..., list[list[float]]]
CompleteFn = Callable[..., Any]
ShapeFn = Callable[[QuerySpec], str]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


def _get(policy: Any, *path: str, default: Any = None) -> Any:
    """Read `policy.a.b` from a pydantic model, Track C's lenient view, or a plain dict."""
    cur = policy
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, cur.get(key.rstrip("_")))
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def inject_text(lessons: list[Lesson], max_tokens: int) -> str:
    """Lessons as a bullet list within a token cap (~4 chars per token), best match first."""
    budget, out = max(0, max_tokens) * 4, []
    for lesson in lessons:
        line = f"- {lesson.text}"
        if sum(map(len, out)) + len(line) > budget:
            break
        out.append(line)
    return "\n".join(out)


class MemoryStore:
    def __init__(
        self,
        db: Database,
        embed: EmbedFn,
        complete: CompleteFn | None = None,
        shape_fn: ShapeFn | None = None,
        model: str = "",
        leak_terms: LeakTerms | None = None,
        search: str = "auto",  # auto: Atlas $vectorSearch, exact cosine scan if unavailable
    ) -> None:
        self.coll = db[MEMORIES]
        self._embed = embed
        self._complete = complete
        self._shape = shape_fn
        self.model = model
        self.leak_terms = leak_terms
        self.search = search
        self._frozen = 0
        self.last_search = ""  # "atlas" | "exact", for `ratchet memory` and tests

    # -- freezing ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        return self._frozen > 0

    @contextmanager
    def frozen(self) -> Iterator[None]:
        """No lesson, counter or status may change inside this block."""
        self._frozen += 1
        try:
            yield
        finally:
            self._frozen -= 1

    def _check_writable(self, what: str) -> None:
        if self.is_frozen:
            raise MemoryFrozenError(f"memory is frozen (holdout/canary evaluation): {what}")

    # -- embedding -----------------------------------------------------------------

    def _embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _embed_query(self, text: str) -> list[float]:
        try:
            return self._embed([text], input_type="query")[0]
        except TypeError:  # embedders without input types
            return self._embed([text])[0]

    # -- retrieve ------------------------------------------------------------------

    def active(self) -> list[Lesson]:
        return [Lesson.model_validate(d) for d in self.coll.find({"status": "active"})]

    def all(self) -> list[Lesson]:
        return [Lesson.model_validate(d) for d in self.coll.find().sort("created_at", 1)]

    def _search_atlas(self, vector: list[float], k: int) -> list[Lesson]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": max(50, k * 10),
                    "limit": k,
                    "filter": {"status": "active"},
                }
            },
            {"$set": {"_vs": {"$meta": "vectorSearchScore"}}},
        ]
        out = []
        for d in self.coll.aggregate(pipeline):
            # Atlas normalises cosine to (1 + cos) / 2; policies are written in plain cosine.
            sim = 2 * float(d.pop("_vs")) - 1
            out.append(Lesson.model_validate(d).model_copy(update={"similarity": sim}))
        return out

    def _search_exact(self, vector: list[float], k: int) -> list[Lesson]:
        scored = [
            x.model_copy(update={"similarity": cosine(vector, x.embedding)}) for x in self.active()
        ]
        scored.sort(key=lambda x: (-(x.similarity or 0.0), x.id))
        return scored[:k]

    def recall(self, shape: str, policy: Any) -> list[Lesson]:
        """Up to `retrieve.k` active lessons whose shape is at least `min_similarity` close.
        Read-only: safe (and allowed) while frozen."""
        k = int(_get(policy, "retrieve", "k", default=0))
        min_sim = float(_get(policy, "retrieve", "min_similarity", default=0.75))
        if k <= 0 or not shape:
            return []
        vector = self._embed_query(shape)
        hits: list[Lesson] | None = None
        if self.search in ("auto", "atlas"):
            try:
                hits = self._search_atlas(vector, k)
                self.last_search = "atlas"
            except Exception:
                if self.search == "atlas":
                    raise
        if hits is None:
            hits = self._search_exact(vector, k)
            self.last_search = "exact"
        return [h for h in hits if (h.similarity or 0.0) >= min_sim - _EPS]

    # -- write ---------------------------------------------------------------------

    @staticmethod
    def _source(r: TuneResult) -> LessonSource | None:
        if r.score.violations:
            return "train_violation"
        if r.score.improvement > 0 and r.score.score > 0:
            return "train_improved"
        return None

    def _distil(self, r: TuneResult, source: LessonSource, queries: dict[str, str], max_chars: int):
        if self._complete is None:
            return None
        brief = {
            "outcome": source,
            "score": round(r.score.score, 4),
            "improvement": round(r.score.improvement, 4),
            "violations": r.score.violations,
            "query_shapes": queries,
            "findings": [
                {"query_id": f.query_id, "detector": f.detector, "hint": f.hint} for f in r.findings
            ],
            "plan": jsonable(r.plan, exclude_none=True),
        }
        messages = [
            {"role": "system", "content": SYSTEM.format(max_chars=max_chars)},
            {"role": "user", "content": json.dumps(brief, default=str)},
        ]
        schema = {
            "type": "object",
            "properties": {"query_id": {"type": "string"}, "lesson": {"type": "string"}},
            "required": ["query_id", "lesson"],
        }
        try:
            resp = self._complete(ROLE, self.model, messages, json_schema=schema)
        except Exception as e:
            if any(c.__name__ == "LLMError" for c in type(e).__mro__):
                return None  # no lesson is better than a crashed generation
            raise
        data = getattr(resp, "data", None)
        if not isinstance(data, dict):
            try:
                data = json.loads(getattr(resp, "text", "") or "")
            except ValueError:
                return None
        text = data.get("lesson") if isinstance(data, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
        return str(data.get("query_id") or ""), " ".join(text.split())[:max_chars]

    def write_lessons(
        self,
        results: list[TuneResult],
        policy: Any,
        queries: dict[str, list[QuerySpec]] | None = None,
    ) -> list[Lesson]:
        """Distil and store lessons from TRAIN results. `queries` maps workload id → its queries
        (for shapes). Returns the lessons that were newly stored (not merged, not dropped)."""
        self._check_writable("write_lessons")
        bad = sorted({f"{r.workload_id} ({r.split})" for r in results if r.split != "train"})
        if bad:
            raise MemoryFrozenError(
                f"lessons may only be written from train runs: {', '.join(bad)}"
            )
        if not _get(policy, "write", "enabled", default=False):
            return []
        allowed = set(_get(policy, "write", "from_", default=[]) or [])
        max_chars = int(_get(policy, "write", "lesson_max_chars", default=300))
        merge_sim = float(_get(policy, "consolidate", "merge_similarity", default=0.92))
        stored: list[Lesson] = []
        for r in results:
            source = self._source(r)
            if source is None or source not in allowed:
                continue
            shapes = {
                q.id: self._shape(q) for q in (queries or {}).get(r.workload_id, []) if self._shape
            }
            got = self._distil(r, source, shapes, max_chars)
            if got is None:
                continue
            qid, text = got
            if self.leak_terms is not None and self.leak_terms.find(text):
                continue  # HOLDOUT_LEAKAGE: never stored
            shape = shapes.get(qid) or next(iter(shapes.values()), "")
            if not shape:
                continue
            shape_vec, text_vec = self._embed_docs([shape, text])
            dup = self._find_duplicate(shape_vec, text_vec, merge_sim)
            if dup is not None:
                self.coll.update_one({"_id": dup.id}, {"$inc": {"support": 1}})
                continue
            digest = hashlib.sha256(f"{shape}\n{text}".encode()).hexdigest()[:12]
            lesson = Lesson(
                id=f"m_{digest}",
                text=text,
                shape=shape,
                embedding=shape_vec,
                text_embedding=text_vec,
                source=source,
                source_workload=r.workload_id,
                harness_version=r.harness_version,
            )
            self.coll.replace_one({"_id": lesson.id}, lesson.to_doc(), upsert=True)
            stored.append(lesson)
        self.consolidate(policy)
        return stored

    def _find_duplicate(self, shape_vec, text_vec, merge_sim: float) -> Lesson | None:
        for x in self.active():
            if (
                cosine(shape_vec, x.embedding) >= merge_sim - _EPS
                and cosine(text_vec, x.text_embedding) >= merge_sim - _EPS
            ):
                return x
        return None

    # -- consolidate / invalidate --------------------------------------------------

    def consolidate(self, policy: Any) -> dict[str, int]:
        """Merge near-duplicate active lessons, then evict the weakest beyond `max_items`."""
        self._check_writable("consolidate")
        merge_sim = float(_get(policy, "consolidate", "merge_similarity", default=0.92))
        max_items = int(_get(policy, "consolidate", "max_items", default=40))

        def strength(x: Lesson) -> tuple:
            return (x.helpful - x.harmful, x.support, -x.created_at.timestamp())

        lessons = sorted(self.active(), key=strength, reverse=True)
        kept: list[Lesson] = []
        merged = 0
        for x in lessons:
            winner = next(
                (
                    k
                    for k in kept
                    if cosine(x.embedding, k.embedding) >= merge_sim - _EPS
                    and cosine(x.text_embedding, k.text_embedding) >= merge_sim - _EPS
                ),
                None,
            )
            if winner is None:
                kept.append(x)
                continue
            self.coll.update_one(
                {"_id": winner.id},
                {"$inc": {"support": x.support, "helpful": x.helpful, "harmful": x.harmful}},
            )
            self.coll.update_one(
                {"_id": x.id}, {"$set": {"status": "merged", "merged_into": winner.id}}
            )
            merged += 1
        evicted = 0
        for x in kept[max_items:]:
            self.coll.update_one({"_id": x.id}, {"$set": {"status": "evicted"}})
            evicted += 1
        return {"merged": merged, "evicted": evicted}

    def record_outcomes(
        self, results: list[TuneResult], baseline: dict[str, float], policy: Any, tol: float = 0.01
    ) -> dict[str, int]:
        """Helpful/harmful accounting for the lessons a TRAIN run used, against the parent's
        score on the same workload; lessons harmful too often are invalidated."""
        self._check_writable("record_outcomes")
        limit = int(_get(policy, "invalidate", "after_harmful_uses", default=2))
        counts = {"helpful": 0, "harmful": 0, "invalidated": 0}
        for r in results:
            if r.split != "train" or not r.lessons_used:
                continue
            base = baseline.get(r.workload_id)
            if r.score.violations or (base is not None and r.score.score < base - tol):
                field = "harmful"
            elif base is not None and r.score.score > base + tol:
                field = "helpful"
            else:
                field = None
            for text in dict.fromkeys(r.lessons_used):
                inc = {"uses": 1, **({field: 1} if field else {})}
                res = self.coll.update_many({"text": text, "status": "active"}, {"$inc": inc})
                if field and res.matched_count:
                    counts[field] += 1
        res = self.coll.update_many(
            {"status": "active", "harmful": {"$gte": limit}}, {"$set": {"status": "invalidated"}}
        )
        counts["invalidated"] = res.modified_count
        return counts

    # -- reporting -----------------------------------------------------------------

    def stats(self) -> dict:
        """Aggregate numbers for the proposer: counts only, no lesson text or shapes."""
        by_status: dict[str, int] = {}
        uses = helpful = harmful = 0
        for d in self.coll.find({}, {"status": 1, "uses": 1, "helpful": 1, "harmful": 1}):
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
            uses += d.get("uses", 0)
            helpful += d.get("helpful", 0)
            harmful += d.get("harmful", 0)
        return {
            "lessons": by_status,
            "active": by_status.get("active", 0),
            "uses": uses,
            "helpful": helpful,
            "harmful": harmful,
        }

    def active_texts(self) -> list[str]:
        return [d["text"] for d in self.coll.find({"status": "active"}, {"text": 1})]
