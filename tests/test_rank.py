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


def test_doubt_aware_description_carries_corrections():
    from backend.recall.rerank import doubt_aware_description

    desc = doubt_aware_description(
        "three red buses on a city street",
        ["city street"],
        [("red buses", "brown buses")],
    )
    assert desc.startswith("three red buses")
    assert "may be wrong about: city street" in desc
    assert "'red buses' to 'brown buses'" in desc
    assert doubt_aware_description("a dog", [], []) == "a dog"


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


def test_weighted_rerank_never_lowers_a_score():
    from backend.recall.rank import Candidate, weighted_rerank

    cands = [
        Candidate("a", 0.30, ["original"], {"original": 0.30}),
        Candidate("b", 0.90, ["original", "drop:0"],
                  {"original": 0.90, "drop:0": 0.10}),
    ]
    out = weighted_rerank(cands, {"drop:0": 1.6})
    by_id = {c.image_id: c for c in out}
    # 'a' is not in the boosted branch: its score must not fall,
    # even though 'b' may pass it.
    assert by_id["a"].score >= 0.30
    assert by_id["b"].score >= 0.90


def test_union_records_rrf_fused_support():
    enc = StubEncoder()
    index = _fixture_index(enc)
    clues = parse_fallback("orange jacket; beach; daylight")
    hyps = hypotheses(clues)
    queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
    union = branch_union(index, queries, per_branch_k=3)
    assert all(c.fused > 0 for c in union)
    multi = max(union, key=lambda c: len(c.branches))
    assert multi.fused >= min(c.fused for c in union)


def test_colour_question_offers_full_palette():
    from backend.recall.clues import Clue
    from backend.recall.questions import attribute_template

    q = attribute_template(Clue("purple bus", "colour", "bus"), 0, 1)
    for colour in ("purple", "brown", "black", "white", "gray"):
        assert colour in q.options


def test_providers_disabled_without_key(monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    from backend.recall.providers import MuseClient, ProviderDisabled, providers_enabled

    assert providers_enabled() is False
    try:
        MuseClient()
        raise AssertionError("expected ProviderDisabled")
    except ProviderDisabled:
        pass


def test_muse_model_comes_from_env(monkeypatch):
    from backend.recall.providers import MUSE_MODEL, MuseClient

    monkeypatch.setenv("META_MODEL_API_KEY", "test-key")
    monkeypatch.delenv("MUSE_MODEL", raising=False)
    assert MuseClient().model == MUSE_MODEL
    monkeypatch.setenv("MUSE_MODEL", "muse-spark-1.3-contributor")
    assert MuseClient().model == "muse-spark-1.3-contributor"
    assert MuseClient("explicit").model == "explicit"
