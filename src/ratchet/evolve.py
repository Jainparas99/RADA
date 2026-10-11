"""Outer loop: the harness evolves, and a deterministic gate decides what is promoted.

Per generation: canary re-check of the head (auto-rollback on failure) → proposer patch from
TRAIN traces → static gate → run the candidate on train + holdout + canary (memory frozen
outside train) → dynamic gate → promote or reject. Everything is recorded in Atlas.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from pymongo.database import Database

from ratchet.config import load_yaml
from ratchet.contracts import GateDecision, PatchOp, ProposerOutput, Split, TuneResult
from ratchet.db import GATE_DECISIONS, GENERATIONS
from ratchet.gate.dynamic import RunMetrics, check_dynamic, metrics_from_results
from ratchet.gate.leakage import LeakTerms
from ratchet.gate.static import static_report
from ratchet.harness import runner
from ratchet.harness.models import HarnessDoc, Scores
from ratchet.harness.store import HarnessStore
from ratchet.proposer.context import ProposerContext, build_context, gate_feedback
from ratchet.proposer.propose import ProposerError

ProposeFn = Callable[[ProposerContext], ProposerOutput]
SPLITS: tuple[Split, ...] = ("train", "holdout", "canary")


class GenerationRecord(BaseModel):
    gen: int
    head_before: str
    candidate: str | None = None
    verdict: str  # ACCEPT | REJECT | NO_PROPOSAL
    reason_codes: list[str] = []
    head_after: str = ""
    rolled_back_from: str | None = None
    rationale: str = ""
    parent_metrics: dict | None = None
    candidate_metrics: dict | None = None
    proposer: dict = {}
    duration_s: float = 0.0
    started_at: datetime


class Evolver:
    def __init__(
        self,
        db: Database,
        backend: runner.Backend,
        propose: ProposeFn,
        memory: Any | None = None,
        proposer_model: str = "",
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.db = db
        self.backend = backend
        self.propose = propose
        self.memory = memory
        self.proposer_model = proposer_model
        self.store = HarnessStore(db)
        self.on_event = on_event or (lambda kind, data: None)
        self.gate_cfg = load_yaml("gate.yaml")
        self.bounds = load_yaml("bounds.yaml")
        leak = self.gate_cfg.get("static", {}).get("leakage", {})
        self.leak_terms = LeakTerms.from_workloads(
            backend.workloads(),
            min_term_chars=leak.get("min_term_chars", 3),
            squash_min_chars=leak.get("squash_min_chars", 6),
            ignore_terms=leak.get("ignore_terms", []),
        )
        if memory is not None and getattr(memory, "leak_terms", None) is None:
            memory.leak_terms = self.leak_terms
        self._fresh: set[tuple[str, str]] = set()  # (version, split) actually run by this process

    # -- running a version ---------------------------------------------------------

    def results_for(self, doc: HarnessDoc, split: Split, rerun: bool = False) -> list[TuneResult]:
        """Stored results for (version, split) if complete, else run and store them."""
        if not rerun:
            have = runner.load_results(self.db, doc.id, split)
            want = {w.id for w in self.backend.workloads(split)}
            if want and {r.workload_id for r in have} == want:
                return have
        self.on_event("run", {"version": doc.id, "split": split})
        self._fresh.add((doc.id, split))
        if split == "train" or self.memory is None:
            return runner.evaluate(self.backend, doc.spec, doc.id, split, db=self.db)
        with self.memory.frozen():  # CLAUDE.md invariant 7
            return runner.evaluate(self.backend, doc.spec, doc.id, split, db=self.db)

    def learn(self, doc: HarnessDoc, train: list[TuneResult]) -> None:
        """Write lessons from a version's TRAIN results (only versions that are, or just
        became, the head)."""
        if self.memory is None:
            return
        queries = {w.id: w.queries for w in self.backend.workloads("train")}
        new = self.memory.write_lessons(train, doc.spec.memory_policy, queries)
        self.on_event("lessons", {"version": doc.id, "written": len(new)})

    def metrics_for(self, doc: HarnessDoc) -> tuple[RunMetrics, dict[Split, list[TuneResult]]]:
        runs = {s: self.results_for(doc, s) for s in SPLITS}
        m = metrics_from_results(runs["train"], runs["holdout"], runs["canary"])
        self.store.set_scores(
            doc.id,
            Scores(
                train=m.train,
                holdout=m.holdout,
                canary_pass=m.canary_pass,
                tokens_per_workload=m.tokens_per_workload,
                violations=m.violations,
            ),
        )
        return m, runs

    # -- canary re-check -----------------------------------------------------------

    def canary_recheck(self) -> str | None:
        """Re-run the head on canaries. On a violation roll back to the nearest accepted
        ancestor that passes. Returns the version rolled back from, if any."""
        start = self.store.get_head()
        doc: HarnessDoc | None = start
        first = True
        while doc is not None:
            results = self.results_for(doc, "canary", rerun=first)
            first = False
            if not any(r.score.violations for r in results):
                break
            self.on_event("canary_failed", {"version": doc.id})
            parent = self.store.get(doc.parent) if doc.parent else None
            while parent is not None and parent.status != "accepted":
                parent = self.store.get(parent.parent) if parent.parent else None
            doc = parent
        if doc is None:
            raise RuntimeError(
                f"head {start.id} fails canaries and has no accepted ancestor that passes"
            )
        if doc.id != start.id:
            self.store.rollback(doc.id, reason="canary")
            self.on_event("rollback", {"from": start.id, "to": doc.id})
            return start.id
        return None

    # -- one generation ------------------------------------------------------------

    def gate_history(self) -> list:
        out = []
        cur = self.db[GATE_DECISIONS].find({}, {"_id": 0}).sort("at", -1).limit(5)
        for d in reversed(list(cur)):
            patch = [PatchOp.model_validate(p) for p in d.pop("patch", [])]
            d.pop("at", None)
            out.append(gate_feedback(GateDecision.model_validate(d), patch))
        return out

    def _models(self, doc: HarnessDoc) -> dict[str, str]:
        routing = doc.spec.model_routing
        return {
            "planner": routing.planner,
            "critic": routing.critic,
            "proposer": self.proposer_model,
        }

    def _record_decision(self, decision: GateDecision, patch: list[PatchOp]) -> None:
        self.db[GATE_DECISIONS].insert_one(
            {
                **decision.model_dump(mode="json"),
                "patch": [p.model_dump(mode="json") for p in patch],
                "at": datetime.now(UTC),
            }
        )

    def next_gen(self) -> int:
        last = self.db[GENERATIONS].find_one(sort=[("gen", -1)])
        return (last["gen"] + 1) if last else 1

    def run_generation(self) -> GenerationRecord:
        t0 = time.monotonic()
        rec = GenerationRecord(
            gen=self.next_gen(), head_before="", verdict="REJECT", started_at=datetime.now(UTC)
        )
        self.on_event("gen_start", {"gen": rec.gen})
        rec.rolled_back_from = self.canary_recheck()
        head = self.store.get_head()
        rec.head_before = rec.head_after = head.id

        parent_metrics, parent_runs = self.metrics_for(head)
        rec.parent_metrics = parent_metrics.model_dump()
        if (head.id, "train") in self._fresh:
            self._fresh.discard((head.id, "train"))
            self.learn(head, parent_runs["train"])

        context = build_context(
            head,
            parent_runs["train"],
            gate_history=self.gate_history(),
            memory_stats=self.memory.stats() if self.memory is not None else {},
            leak_terms=self.leak_terms,
            bounds=self.bounds,
            gate_cfg=self.gate_cfg,
        )
        try:
            out = self.propose(context)
        except ProposerError as e:
            rec.verdict, rec.rationale = "NO_PROPOSAL", str(e)[:500]
            return self._finish(rec, t0)
        rec.rationale = out.hypothesis
        rationale = (
            f"{out.diagnosis}\n\nHypothesis: {out.hypothesis}\n\nExpected: {out.expected_effect}"
        )
        self.on_event("proposal", {"patch": [p.model_dump() for p in out.patch], "why": rationale})

        static = static_report(
            head.spec,
            out.patch,
            leak_terms=self.leak_terms,
            lessons=self.memory.active_texts() if self.memory is not None else None,
            bounds=self.bounds,
            gate_cfg=self.gate_cfg,
        )
        if not static.ok or static.candidate is None:
            # The patch never becomes a harness: store it against the unchanged parent spec.
            cand = self.store.save_candidate(
                head.id, head.spec, out.patch, rationale, models_used=self._models(head)
            )
            decision = GateDecision(
                candidate=cand.id,
                parent=head.id,
                verdict="REJECT",
                reason_codes=static.codes,
                metrics={
                    "stage": "static",
                    "details": static.details,
                    "parent": rec.parent_metrics,
                },
            )
            return self._decide(rec, decision, out.patch, t0)

        cand = self.store.save_candidate(
            head.id,
            static.candidate,
            out.patch,
            rationale,
            models_used={**self._models(head), **static.candidate.model_routing.model_dump()},
        )
        cand_metrics, cand_runs = self.metrics_for(cand)
        rec.candidate_metrics = cand_metrics.model_dump()
        if self.memory is not None:
            baseline = {r.workload_id: r.score.score for r in parent_runs["train"]}
            self.memory.record_outcomes(cand_runs["train"], baseline, cand.spec.memory_policy)
        codes = check_dynamic(
            parent_metrics,
            cand_metrics,
            cfg=self.gate_cfg,
            token_budget_per_workload=head.spec.protected.token_budget_per_workload,
        )
        decision = GateDecision(
            candidate=cand.id,
            parent=head.id,
            verdict="REJECT" if codes else "ACCEPT",
            reason_codes=codes,
            metrics={
                "stage": "dynamic",
                "parent": rec.parent_metrics,
                "candidate": rec.candidate_metrics,
            },
        )
        rec = self._decide(rec, decision, out.patch, t0, finish=False)
        if decision.verdict == "ACCEPT":
            self._fresh.discard((cand.id, "train"))
            self.learn(self.store.get(cand.id), cand_runs["train"])
        return self._finish(rec, t0)

    def _decide(
        self, rec, decision: GateDecision, patch: list[PatchOp], t0: float, finish: bool = True
    ):
        rec.candidate, rec.verdict, rec.reason_codes = (
            decision.candidate,
            decision.verdict,
            decision.reason_codes,
        )
        if decision.verdict == "ACCEPT":
            self.store.promote(decision.candidate)
            rec.head_after = decision.candidate
        else:
            self.store.reject(decision.candidate, decision.reason_codes)
        self._record_decision(decision, patch)
        return self._finish(rec, t0) if finish else rec

    def _finish(self, rec: GenerationRecord, t0: float) -> GenerationRecord:
        rec.duration_s = round(time.monotonic() - t0, 3)
        self.db[GENERATIONS].insert_one(rec.model_dump(mode="python"))
        self.on_event("gen_end", rec.model_dump(mode="json"))
        return rec

    def run(self, gens: int) -> list[GenerationRecord]:
        return [self.run_generation() for _ in range(gens)]


def evolve(gens: int) -> None:
    """SPEC §10 entry point: real backend, real proposer, the `ratchet_<ns>` database."""
    from ratchet import memory
    from ratchet.config import get_settings
    from ratchet.db import get_db
    from ratchet.proposer.propose import propose

    model = get_settings().proposer_model
    mem = memory.default_store()  # the same store the agent recalls from
    Evolver(get_db(), runner.RealBackend(), propose, memory=mem, proposer_model=model).run(gens)
