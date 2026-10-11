"""`ratchet watch`: live tail of harness versions, gate decisions and generations (SPEC §13).

Uses a MongoDB change stream on the state database when the tier supports it, else polls.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

from ratchet.db import GATE_DECISIONS, GENERATIONS, HARNESS_VERSIONS

WATCHED = (HARNESS_VERSIONS, GATE_DECISIONS, GENERATIONS)


def _f(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, int | float) else "-"


def format_event(coll: str, doc: dict) -> str:
    """One Rich-markup line per event."""
    if coll == GATE_DECISIONS:
        verdict = doc.get("verdict", "?")
        colour = "green" if verdict == "ACCEPT" else "red"
        codes = ", ".join(doc.get("reason_codes") or []) or "-"
        return (
            f"[b]gate[/b] {doc.get('candidate', '?')} vs {doc.get('parent', '?')}: "
            f"[{colour}]{verdict}[/{colour}] {codes}"
        )
    if coll == HARNESS_VERSIONS:
        s = doc.get("scores") or {}
        return (
            f"[b]version[/b] {doc.get('_id', '?')} [{doc.get('status', '?')}] "
            f"train {_f(s.get('train'))} holdout {_f(s.get('holdout'))} "
            f"{(doc.get('rationale') or '')[:80]}"
        )
    if coll == GENERATIONS:
        return (
            f"[b]gen {doc.get('gen', '?')}[/b] {doc.get('head_before', '?')} -> "
            f"{doc.get('head_after', '?')}: {doc.get('verdict', '?')} "
            f"{', '.join(doc.get('reason_codes') or [])} ({_f(doc.get('duration_s'))}s)"
        )
    return f"{coll}: {doc.get('_id', '?')}"


def _key(coll: str, doc: dict) -> tuple:
    # versions change status in place (candidate -> accepted/rejected): treat that as new
    return (coll, str(doc.get("_id")), doc.get("status"), str(doc.get("scores")))


def poll_once(db: Any, seen: set[tuple], limit: int = 50) -> list[tuple[str, dict]]:
    new: list[tuple[str, dict]] = []
    for coll in WATCHED:
        docs = list(db[coll].find({}).sort("_id", -1).limit(limit))
        for doc in reversed(docs):
            k = _key(coll, doc)
            if k not in seen:
                seen.add(k)
                new.append((coll, doc))
    return new


def change_stream_events(db: Any) -> Iterator[tuple[str, dict]]:
    pipeline = [
        {
            "$match": {
                "ns.coll": {"$in": list(WATCHED)},
                "operationType": {"$in": ["insert", "update", "replace"]},
            }
        }
    ]
    with db.watch(pipeline=pipeline, full_document="updateLookup") as stream:
        for change in stream:
            yield change["ns"]["coll"], change.get("fullDocument") or {}


def watch(
    db: Any,
    printer: Callable[[str], None],
    interval: float = 2.0,
    use_change_streams: bool = True,
    max_events: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_polls: int | None = None,
) -> str:
    """Tail events until interrupted (or `max_events`). Returns the mode used."""
    count = 0
    if use_change_streams:
        try:
            printer("[dim]watching via change streams (Ctrl-C to stop)[/dim]")
            for coll, doc in change_stream_events(db):
                printer(format_event(coll, doc))
                count += 1
                if max_events is not None and count >= max_events:
                    return "change_stream"
            return "change_stream"
        except Exception as e:  # unsupported on tier / permissions / mongomock
            printer(f"[yellow]change streams unavailable ({type(e).__name__}); polling[/yellow]")
    seen: set[tuple] = set()
    for _coll, _doc in poll_once(db, seen):  # existing history: print the last few only
        pass
    polls = 0
    while max_polls is None or polls < max_polls:
        for coll, doc in poll_once(db, seen):
            printer(format_event(coll, doc))
            count += 1
            if max_events is not None and count >= max_events:
                return "poll"
        polls += 1
        sleep(interval)
    return "poll"
