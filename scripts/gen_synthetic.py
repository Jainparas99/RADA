"""Generate the seeded synthetic collection ratchet_data.orders (SPEC §5.1).

Deterministic: the same --seed and --n always produce byte-identical documents, so the
holdout workloads on it are reproducible for every teammate. Idempotent: if the
collection already holds --n docs generated with the same seed, nothing is written.

    uv run python scripts/gen_synthetic.py            # 200k docs, seed 7
    uv run python scripts/gen_synthetic.py --n 50000  # smaller, if storage is tight
"""

from __future__ import annotations

import argparse
import datetime as dt
import random

from ratchet.env import conn

DB, COLL = "ratchet_data", "orders"
REGIONS = [
    "us-east",
    "us-west",
    "eu-west",
    "eu-central",
    "ap-south",
    "ap-east",
    "sa-east",
    "af-south",
]
STATUSES = (
    ["delivered"] * 60 + ["shipped"] * 15 + ["pending"] * 10 + ["cancelled"] * 8 + ["returned"] * 7
)
CHANNELS = ["web"] * 55 + ["app"] * 35 + ["store"] * 10
COUPONS = ["SAVE10", "SPRING", "VIP", "WELCOME", "FREESHIP"]
START = dt.datetime(2025, 1, 1)
SPAN_S = 2 * 365 * 24 * 3600


def make_doc(i: int, rnd: random.Random) -> dict:
    n_items = 1 + int(rnd.random() * 4)
    items = [
        {"sku": f"SKU-{int(rnd.random() * 500):04d}", "qty": 1 + int(rnd.random() * 5)}
        for _ in range(n_items)
    ]
    return {
        "_id": i,
        "customer_id": int(rnd.random() * 20_000),
        "status": STATUSES[int(rnd.random() * len(STATUSES))],
        "region": REGIONS[int(rnd.random() * len(REGIONS))],
        "channel": CHANNELS[int(rnd.random() * len(CHANNELS))],
        "priority": rnd.random() < 0.05,
        "created_at": START + dt.timedelta(seconds=int(rnd.random() * SPAN_S)),
        "total_cents": min(1_000_000, int(rnd.lognormvariate(8.5, 0.9))),
        "items": items,
        "coupon": COUPONS[int(rnd.random() * len(COUPONS))] if rnd.random() < 0.2 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=5_000)
    args = ap.parse_args()

    db = conn.rw_client()[DB]
    meta = db["_ratchet_meta"]
    m = meta.find_one({"_id": COLL})
    if (
        m
        and m.get("seed") == args.seed
        and m.get("n") == args.n
        and m.get("complete")
        and db[COLL].estimated_document_count() == args.n
    ):
        print(f"{DB}.{COLL} already has {args.n} docs (seed {args.seed}); nothing to do")
        return
    db[COLL].drop()
    meta.replace_one(
        {"_id": COLL}, {"_id": COLL, "seed": args.seed, "n": args.n, "complete": False}, upsert=True
    )
    rnd = random.Random(args.seed)
    batch = []
    for i in range(args.n):
        batch.append(make_doc(i, rnd))
        if len(batch) >= args.batch:
            db[COLL].insert_many(batch, ordered=True)
            batch = []
            print(f"  {i + 1}/{args.n}", end="\r", flush=True)
    if batch:
        db[COLL].insert_many(batch, ordered=True)
    meta.replace_one(
        {"_id": COLL}, {"_id": COLL, "seed": args.seed, "n": args.n, "complete": True}, upsert=True
    )
    size_mb = db.command("collStats", COLL).get("size", 0) / 1e6
    print(f"\nwrote {args.n} docs to {DB}.{COLL} ({size_mb:.1f} MB uncompressed)")


if __name__ == "__main__":
    main()
