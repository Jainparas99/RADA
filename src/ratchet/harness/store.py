"""Harness versions in Atlas: `harness_versions` (one doc per version) and `harness_head`."""

from __future__ import annotations

from datetime import UTC, datetime

from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from ratchet.contracts import PatchOp
from ratchet.db import HARNESS_HEAD, HARNESS_VERSIONS
from ratchet.harness.models import HarnessDoc, HarnessSpec, Scores, version_id

HEAD_ID = "head"


class StoreError(RuntimeError):
    pass


class HarnessStore:
    def __init__(self, db: Database) -> None:
        self.versions = db[HARNESS_VERSIONS]
        self.head = db[HARNESS_HEAD]

    # -- reads ---------------------------------------------------------------------

    def get(self, v: int | str) -> HarnessDoc:
        doc = self.versions.find_one({"_id": version_id(v)})
        if doc is None:
            raise StoreError(f"no harness version {version_id(v)}")
        return HarnessDoc.from_doc(doc)

    def head_id(self) -> str | None:
        doc = self.head.find_one({"_id": HEAD_ID})
        return doc["version_id"] if doc else None

    def get_head(self) -> HarnessDoc:
        hid = self.head_id()
        if hid is None:
            raise StoreError("no harness head; run `ratchet init`")
        return self.get(hid)

    def history(self) -> list[HarnessDoc]:
        return [HarnessDoc.from_doc(d) for d in self.versions.find().sort("version", 1)]

    def head_log(self) -> list[dict]:
        doc = self.head.find_one({"_id": HEAD_ID})
        return list(doc.get("log", [])) if doc else []

    # -- writes --------------------------------------------------------------------

    def init_gen0(self, spec: HarnessSpec) -> HarnessDoc:
        """Create h_0000 and point the head at it. No-op if a head already exists."""
        if self.head_id() is not None:
            return self.get_head()
        doc = self._insert(
            parent=None,
            spec=spec,
            patch=[],
            rationale="gen 0: deliberately minimal baseline",
            created_by="human",
            models_used={},
            status="accepted",
        )
        self._move_head(doc.id, "init")
        return doc

    def save_candidate(
        self,
        parent: int | str,
        spec: HarnessSpec,
        patch: list[PatchOp] | None = None,
        rationale: str = "",
        created_by: str = "proposer",
        models_used: dict[str, str] | None = None,
    ) -> HarnessDoc:
        parent_doc = self.get(parent)
        return self._insert(
            parent=parent_doc.id,
            spec=spec,
            patch=patch or [],
            rationale=rationale,
            created_by=created_by,
            models_used=models_used or {},
            status="candidate",
        )

    def set_scores(self, v: int | str, scores: Scores) -> None:
        self._update(v, {"scores": scores.model_dump()})

    def reject(self, v: int | str, reason_codes: list[str]) -> HarnessDoc:
        doc = self.get(v)
        if doc.status != "candidate":
            raise StoreError(f"{doc.id} is {doc.status}, not a candidate")
        self._update(v, {"status": "rejected", "reason_codes": list(reason_codes)})
        return self.get(v)

    def promote(self, v: int | str) -> HarnessDoc:
        """Accept a candidate and make it the head. Its parent must be the current head."""
        doc = self.get(v)
        if doc.status != "candidate":
            raise StoreError(f"{doc.id} is {doc.status}, not a candidate")
        if doc.parent != self.head_id():
            raise StoreError(f"{doc.id} was built on {doc.parent}, but head is {self.head_id()}")
        self._update(v, {"status": "accepted", "reason_codes": []})
        self._move_head(doc.id, "promote")
        return self.get(v)

    def rollback(self, v: int | str, reason: str = "manual") -> HarnessDoc:
        """Point the head back at an earlier accepted version. History is kept."""
        doc = self.get(v)
        if doc.status != "accepted":
            raise StoreError(f"can only roll back to an accepted version; {doc.id} is {doc.status}")
        self._move_head(doc.id, f"rollback:{reason}")
        return doc

    def repin(self, scorer_ref: dict, gate_ref: dict, rationale: str) -> HarnessDoc:
        """Human-only: new head identical to the current one but pinned to new source hashes."""
        head = self.get_head()
        spec_doc = head.spec.to_doc()
        spec_doc["protected"]["scorer_ref"] = scorer_ref
        spec_doc["protected"]["gate_ref"] = gate_ref
        doc = self._insert(
            parent=head.id,
            spec=HarnessSpec.from_doc(spec_doc),
            patch=[],
            rationale=rationale,
            created_by="human",
            models_used={},
            status="accepted",
        )
        self.set_scores(doc.id, head.scores)
        self._move_head(doc.id, "pin")
        return self.get(doc.id)

    # -- internals -----------------------------------------------------------------

    def _insert(self, *, parent, spec, patch, rationale, created_by, models_used, status):
        for _ in range(5):
            last = self.versions.find_one(sort=[("version", -1)])
            n = (last["version"] + 1) if last else 0
            doc = HarnessDoc(
                id=version_id(n),
                version=n,
                parent=parent,
                status=status,
                created_by=created_by,
                rationale=rationale,
                patch=patch,
                spec=spec,
                models_used=models_used,
            )
            try:
                self.versions.insert_one(doc.to_doc())
                return doc
            except DuplicateKeyError:
                continue  # another writer took this version number
        raise StoreError("could not allocate a harness version")

    def _update(self, v: int | str, fields: dict) -> None:
        res = self.versions.update_one({"_id": version_id(v)}, {"$set": fields})
        if res.matched_count != 1:
            raise StoreError(f"no harness version {version_id(v)}")

    def _move_head(self, to: str, action: str) -> None:
        now = datetime.now(UTC)
        entry = {"from": self.head_id(), "to": to, "action": action, "at": now}
        self.head.update_one(
            {"_id": HEAD_ID},
            {"$set": {"version_id": to, "updated_at": now}, "$push": {"log": entry}},
            upsert=True,
        )


def _default() -> HarnessStore:
    from ratchet.db import get_db

    return HarnessStore(get_db())


# Module-level interface from SPEC §10, bound to the `ratchet_<ns>` database.
def get_head() -> HarnessDoc:
    return _default().get_head()


def save_candidate(parent: int | str, spec: HarnessSpec, **kwargs) -> HarnessDoc:
    return _default().save_candidate(parent, spec, **kwargs)


def promote(v: int | str) -> HarnessDoc:
    return _default().promote(v)


def rollback(v: int | str) -> HarnessDoc:
    return _default().rollback(v)


def history() -> list[HarnessDoc]:
    return _default().history()
