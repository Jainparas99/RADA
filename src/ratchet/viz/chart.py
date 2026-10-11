"""`ratchet chart`: PNG of harness score per generation (SPEC §13).

Head holdout (solid) and head train (dashed) as step lines; every candidate's holdout score as a
point: accepted ● green, rejected ✕ red with its first reason code. Data comes from `generations`
(Track A's evolve records), falling back to `harness_versions`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class GenPoint(BaseModel):
    gen: int
    verdict: str  # ACCEPT | REJECT | NO_PROPOSAL
    candidate: str | None = None
    reason_codes: list[str] = []
    cand_train: float | None = None
    cand_holdout: float | None = None
    head_train: float | None = None
    head_holdout: float | None = None


def _m(d: Any, key: str) -> float | None:
    if isinstance(d, dict) and isinstance(d.get(key), int | float):
        return float(d[key])
    return None


def points_from_generations(docs: list[dict]) -> list[GenPoint]:
    """One point per generation record, plus gen 0 from the first record's parent metrics."""
    docs = sorted(docs, key=lambda d: d.get("gen", 0))
    out: list[GenPoint] = []
    if docs and isinstance(docs[0].get("parent_metrics"), dict):
        pm = docs[0]["parent_metrics"]
        out.append(
            GenPoint(
                gen=0,
                verdict="ACCEPT",
                candidate=docs[0].get("head_before"),
                head_train=_m(pm, "train"),
                head_holdout=_m(pm, "holdout"),
            )
        )
    head_train = out[0].head_train if out else None
    head_holdout = out[0].head_holdout if out else None
    for d in docs:
        pm, cm = d.get("parent_metrics") or {}, d.get("candidate_metrics") or {}
        verdict = str(d.get("verdict", "?"))
        if _m(pm, "holdout") is not None:
            head_train, head_holdout = _m(pm, "train"), _m(pm, "holdout")
        if verdict == "ACCEPT" and _m(cm, "holdout") is not None:
            head_train, head_holdout = _m(cm, "train"), _m(cm, "holdout")
        out.append(
            GenPoint(
                gen=int(d.get("gen", 0)),
                verdict=verdict,
                candidate=d.get("candidate"),
                reason_codes=list(d.get("reason_codes") or []),
                cand_train=_m(cm, "train"),
                cand_holdout=_m(cm, "holdout"),
                head_train=head_train,
                head_holdout=head_holdout,
            )
        )
    return out


def points_from_versions(docs: list[dict]) -> list[GenPoint]:
    """Fallback: harness_versions in version order; accepted versions move the head."""
    out: list[GenPoint] = []
    head_t = head_h = None
    for d in sorted(docs, key=lambda d: d.get("version", 0)):
        s = d.get("scores") or {}
        accepted = d.get("status") == "accepted"
        if accepted:
            head_t, head_h = _m(s, "train"), _m(s, "holdout")
        out.append(
            GenPoint(
                gen=int(d.get("version", 0)),
                verdict="ACCEPT" if accepted else "REJECT",
                candidate=d.get("_id"),
                reason_codes=list(d.get("reason_codes") or []),
                cand_train=_m(s, "train"),
                cand_holdout=_m(s, "holdout"),
                head_train=head_t,
                head_holdout=head_h,
            )
        )
    return out


def load_points(db: Any) -> list[GenPoint]:
    from ratchet.db import GENERATIONS, HARNESS_VERSIONS

    gens = list(db[GENERATIONS].find({}, {"_id": 0}))
    if gens:
        return points_from_generations(gens)
    return points_from_versions(list(db[HARNESS_VERSIONS].find({})))


def render_chart(points: list[GenPoint], out: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    head = [p for p in points if p.head_holdout is not None]
    if head:
        xs = [p.gen for p in head]
        ax.step(
            xs,
            [p.head_holdout for p in head],
            where="post",
            lw=2.4,
            color="#1f6feb",
            label="head: holdout (unseen workloads)",
        )
        if any(p.head_train is not None for p in head):
            ax.step(
                xs,
                [p.head_train if p.head_train is not None else float("nan") for p in head],
                where="post",
                lw=1.4,
                ls="--",
                color="#8b949e",
                label="head: train",
            )
    acc = [p for p in points if p.verdict == "ACCEPT" and p.cand_holdout is not None]
    rej = [p for p in points if p.verdict == "REJECT" and p.cand_holdout is not None]
    if acc:
        ax.scatter(
            [p.gen for p in acc],
            [p.cand_holdout for p in acc],
            s=70,
            color="#2da44e",
            zorder=5,
            label="candidate accepted",
        )
    if rej:
        ax.scatter(
            [p.gen for p in rej],
            [p.cand_holdout for p in rej],
            s=80,
            marker="x",
            color="#cf222e",
            linewidths=2.2,
            zorder=5,
            label="candidate rejected",
        )
        for p in rej:
            code = p.reason_codes[0] if p.reason_codes else "REJECT"
            ax.annotate(
                code,
                (p.gen, p.cand_holdout),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=7,
                color="#cf222e",
            )
    for p in acc:
        if p.candidate:
            ax.annotate(
                p.candidate,
                (p.gen, p.cand_holdout),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color="#2da44e",
            )
    ax.set_xlabel("generation")
    ax.set_ylabel("mean workload score (improvement − index penalty)")
    ax.set_title(title or "Ratchet: harness score per generation (holdout gates promotion)")
    if points:
        ax.set_xticks(sorted({p.gen for p in points}))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
