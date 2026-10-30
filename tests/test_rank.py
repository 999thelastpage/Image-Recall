from backend.recall.clues import hypotheses, parse_fallback
from backend.recall.embeddings import StubEncoder
from backend.recall.rank import branch_union
from backend.recall.retrieve import ExactIndex


def _fixture_index(enc: StubEncoder) -> ExactIndex:
    captions = [
        "orange jacket; beach; daylight",  # target: 'red' is wrong
        "red jacket; city; night",
        "blue car; street; daylight",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    return ExactIndex(ids, enc.encode_texts(captions))


def test_dropped_clue_branch_surfaces_target():
    """With 'red' wrong, the drop:0 branch must surface the orange
    jacket in the union (stub vectors: exact text overlap decides)."""
    enc = StubEncoder()
    index = _fixture_index(enc)
    clues = parse_fallback("red jacket; beach; daylight")
    hyps = hypotheses(clues)
    queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
    union = branch_union(index, queries, per_branch_k=3)
    by_id = {c.image_id: c for c in union}
    assert "img-000000" in by_id
    assert "drop:0" in by_id["img-000000"].branches


def test_union_keeps_provenance_and_best_score():
    enc = StubEncoder()
    index = _fixture_index(enc)
    clues = parse_fallback("orange jacket; beach; daylight")
    hyps = hypotheses(clues)
    queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
    union = branch_union(index, queries, per_branch_k=3)
    top = union[0]
    assert top.image_id == "img-000000"
    assert "original" in top.branches
    assert top.score > 0.99


def test_providers_disabled_without_key(monkeypatch):
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    from backend.recall.providers import MuseClient, ProviderDisabled, providers_enabled

    assert providers_enabled() is False
    try:
        MuseClient()
        raise AssertionError("expected ProviderDisabled")
    except ProviderDisabled:
        pass
