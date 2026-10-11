from ratchet.contracts import QuerySpec
from ratchet.env.shape import query_shape


def q(**kw):
    return QuerySpec(id="q", **kw)


def test_spec_example_find():
    s = query_shape(
        q(
            kind="find",
            collection="movies",
            filter={"genres": "Drama", "year": {"$gt": 2000}},
            sort=[("imdb.rating", -1), ("_id", 1)],
            limit=10,
            projection={"title": 1, "imdb.rating": 1},
        )
    )
    assert s == (
        "movies | filter {genres: EQ, year: RANGE} | sort {imdb.rating: DESC, _id: ASC} "
        "| proj {imdb.rating, title} | limit"
    )


def test_values_stripped_and_field_order_irrelevant():
    a = q(kind="find", collection="m", filter={"year": {"$gt": 1}, "genres": "X"})
    b = q(kind="find", collection="m", filter={"genres": "Y", "year": {"$gt": 1999}})
    assert query_shape(a) == query_shape(b)


def test_sort_order_matters():
    a = q(kind="find", collection="m", sort=[("a", 1), ("b", 1)])
    b = q(kind="find", collection="m", sort=[("b", 1), ("a", 1)])
    assert query_shape(a) != query_shape(b)


def test_operators():
    s = query_shape(
        q(
            kind="find",
            collection="m",
            filter={
                "$and": [{"a": {"$in": [1, 2]}}, {"b": {"$regex": "^x"}}],
                "c": {"$regex": "x", "$options": "i"},
                "d": {"$exists": True, "$ne": None},
                "e": {"$elemMatch": {"t": "exam", "s": {"$gte": 1}}},
                "$or": [{"f": 1}, {"g": {"$lt": 2}}],
            },
        )
    )
    assert s == (
        "m | filter {a: IN, b: REGEX_PREFIX, c: REGEX, d: EXISTS+NE, "
        "e: ELEM_MATCH{s: RANGE, t: EQ}, OR[{f: EQ}, {g: RANGE}]}"
    )


def test_spec_example_aggregate():
    s = query_shape(
        q(
            kind="aggregate",
            collection="comments",
            pipeline=[
                {
                    "$lookup": {
                        "from": "movies",
                        "localField": "movie_id",
                        "foreignField": "_id",
                        "as": "m",
                    }
                },
                {"$unwind": "$m"},
                {"$match": {"m.year": 2012}},
                {"$count": "n"},
            ],
        )
    )
    assert s == (
        "comments | pipeline $lookup(from=movies, local=movie_id, foreign=_id, as=m) "
        "> $unwind(m) > $match {m.year: EQ} > $count"
    )


def test_group_and_project():
    s = query_shape(
        q(
            kind="aggregate",
            collection="c",
            pipeline=[
                {"$group": {"_id": "$year", "n": {"$sum": 1}, "best": {"$max": "$x"}}},
                {"$project": {"n": 1, "r": {"$add": ["$n", 1]}, "_id": 0}},
                {"$sort": {"n": -1, "_id": 1}},
            ],
        )
    )
    assert s == (
        "c | pipeline $group(by=year, acc=[max, sum]) > $project {-_id, n, r=EXPR} "
        "> $sort {n: DESC, _id: ASC}"
    )


def test_stable_string():
    a = q(kind="find", collection="m", filter={"b": 1, "a": 2}, projection={"y": 1, "x": 1})
    assert (
        query_shape(a) == query_shape(a.model_copy()) == "m | filter {a: EQ, b: EQ} | proj {x, y}"
    )
