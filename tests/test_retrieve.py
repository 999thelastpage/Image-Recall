import numpy as np

from backend.recall.embeddings import StubEncoder
from backend.recall.retrieve import ExactIndex


def _index():
    enc = StubEncoder(dim=32)
    ids = [f"img-{i:06d}" for i in range(50)]
    mat = enc.encode_texts([f"caption number {i}" for i in range(50)])
    return ExactIndex(ids, mat), enc


def test_exact_match_ranks_first():
    index, enc = _index()
    q = enc.encode_texts(["caption number 7"])
    results = index.search(q, 5)
    assert results[0][0] == "img-000007"
    assert results[0][1] > 0.99


def test_search_is_deterministic_and_sorted():
    index, enc = _index()
    q = enc.encode_texts(["unseen query text"])
    a = index.search(q, 10)
    b = index.search(q, 10)
    assert a == b
    scores = [s for _i, s in a]
    assert scores == sorted(scores, reverse=True)


def test_rank_of_matches_search():
    index, enc = _index()
    q = enc.encode_texts(["caption number 3"])
    assert index.rank_of(q, "img-000003") == 1
    assert index.rank_of(q, "img-999999") is None


def test_row_mismatch_rejected():
    enc = StubEncoder(dim=8)
    try:
        ExactIndex(["a"], enc.encode_texts(["x", "y"]))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_topk_keeps_id_ascending_ties_at_the_cut():
    # scores: z-high .9 | b-tie .5, a-tie .5 | c-low .1 (id order shuffled)
    mat = np.array([[0.9, 0.0], [0.5, 0.0], [0.5, 0.0], [0.1, 0.0]], dtype=np.float32)
    index = ExactIndex(["z-high", "b-tie", "a-tie", "c-low"], mat)
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert [i for i, _s in index.search(q, 2)] == ["z-high", "a-tie"]
    assert [i for i, _s in index.search(q, 3)] == ["z-high", "a-tie", "b-tie"]
    assert [i for i, _s in index.search(q, 99)] == [
        "z-high", "a-tie", "b-tie", "c-low"]


def test_search_many_matches_search_per_query():
    rng = np.random.default_rng(7)
    ids = [f"img-{i:06d}" for i in range(300)]
    mat = rng.standard_normal((300, 24)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    index = ExactIndex(ids, mat)
    queries = rng.standard_normal((5, 24)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    batched = index.search_many(queries, 17)
    assert len(batched) == 5
    for j in range(5):
        solo = index.search(queries[j], 17)
        assert [i for i, _s in batched[j]] == [i for i, _s in solo]
        assert [round(s, 6) for _i, s in batched[j]] == [
            round(s, 6) for _i, s in solo]


def test_branch_quota_scales_with_corpus():
    from backend.recall.config import branch_quota

    assert branch_quota(5_000) == 50
    assert branch_quota(82_783) == 150
    assert branch_quota(500_000) == 250
