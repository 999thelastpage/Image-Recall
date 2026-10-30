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
