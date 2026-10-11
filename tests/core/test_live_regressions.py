"""Bugs from the first live run against Atlas (2026-10-11); none needs a database to reproduce."""

import datetime

from bson import ObjectId

from ratchet import db as dbm
from ratchet.contracts import QuerySpec
from ratchet.env.explain import explain_command, to_bson
from ratchet.gate.leakage import LeakTerms
from ratchet.gate.static import check_static
from ratchet.harness import runner
from ratchet.harness.gen0 import gen0_spec
from ratchet.harness.models import HarnessDoc
from ratchet.jsonsafe import jsonable
from ratchet.proposer.context import build_context
from tests.fakes import FakeBackend

OID = "573a1390f29313caabcd4135"


def test_extended_json_in_workload_queries_reaches_the_server_as_bson():
    """Workload files write ObjectIds and dates as {"$oid"} / {"$date"}; sent as-is the server
    answers `unknown operator: $oid`."""
    find = QuerySpec(id="q", kind="find", collection="c", filter={"_id": {"$oid": OID}}, limit=1)
    assert explain_command(find)["filter"] == {"_id": ObjectId(OID)}
    agg = QuerySpec(
        id="q",
        kind="aggregate",
        collection="c",
        pipeline=[{"$match": {"d": {"$gte": {"$date": "2016-01-01T00:00:00Z"}}}}],
    )
    sent = explain_command(agg)["pipeline"][0]["$match"]["d"]["$gte"]
    assert isinstance(sent, datetime.datetime) and sent.year == 2016
    assert to_bson(to_bson(find.filter)) == to_bson(find.filter)  # idempotent
    assert find.filter == {"_id": {"$oid": OID}}  # the model itself stays plain JSON


def test_every_workload_file_loads_as_plain_json_and_converts():
    for w in runner.load_workload_files():
        for q in w.queries:
            explain_command(q)


def test_results_with_bson_values_can_be_stored_and_shown_to_the_proposer(db, spec):
    """Explain output carries ObjectIds; pydantic's JSON mode cannot serialise them."""
    r = runner.evaluate(FakeBackend(), spec, "h_0000", "train")[0]
    summary = r.before.per_query["q1"].model_copy(update={"raw": {"bound": ObjectId(OID)}})
    r = r.model_copy(
        update={
            "before": r.before.model_copy(update={"per_query": {"q1": summary}}),
            "trace": [{"step": "explain", "seen": ObjectId(OID)}],
        }
    )
    runner.save_results(db, [r])
    back = runner.load_results(db, "h_0000", "train")[0]
    assert back.before.per_query["q1"].raw == {"bound": ObjectId(OID)}
    assert db[dbm.TUNE_RESULTS].count_documents({}) == 1
    head = HarnessDoc(id="h_0000", version=0, status="accepted", spec=spec)
    ctx = build_context(head, [r])
    assert OID in ctx.failures[0].trace
    assert jsonable({"a": ObjectId(OID)}) == {"a": {"$oid": OID}}


def test_an_unpatched_spec_never_trips_the_leak_check_on_the_real_workloads():
    """`memory_policy.consolidate.max_items` contains "items", a held-out field name. The ignore
    list in gate.yaml covers it, and is now the default for every caller."""
    terms = LeakTerms.from_workloads(runner.load_workload_files())
    assert "items" not in terms.terms and "sales" in terms.terms
    assert check_static(gen0_spec(), [], leak_terms=terms) == []
