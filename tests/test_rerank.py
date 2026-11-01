from backend.recall.rank import Candidate
from backend.recall.rerank import MuseReranker, doubt_aware_description


class FakeClaude:
    """Scores by filename so tests are deterministic and offline."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def chat_with_image(self, path: str, prompt: str) -> str:
        import json
        from pathlib import Path

        return json.dumps({"score": self.scores[Path(path).name]})


def _cands() -> list[Candidate]:
    return [
        Candidate("img-000000", 0.30, ["original"], {"original": 0.30}),
        Candidate("img-000001", 0.29, ["drop:0"], {"drop:0": 0.29}),
        Candidate("img-000002", 0.20, ["original"], {"original": 0.20}),
    ]


def test_rerank_promotes_muse_favourite():
    paths = {c.image_id: f"/x/{c.image_id}.jpg" for c in _cands()}
    r = MuseReranker(FakeClaude({"img-000000.jpg": 1, "img-000001.jpg": 10,
                                 "img-000002.jpg": 2}))
    out = r.rerank(_cands(), paths, "a photo", top_n=3, weight=0.45)
    assert out[0].image_id == "img-000001"
    assert out[0].muse == 1.0


def test_doubt_aware_description_without_notes_is_unchanged():
    assert doubt_aware_description("a purple car in grass", [], []) == (
        "a purple car in grass"
    )
    assert doubt_aware_description(
        "a purple car in grass", ["purple car"], [("red", "blue")]
    ) == (
        "a purple car in grass "
        "Note: the user may be wrong about: purple car. "
        "Do not penalise a photo for differing on those details. "
        "The user corrected 'red' to 'blue'."
    )


def test_doubt_aware_description_appends_notes_verbatim():
    out = doubt_aware_description(
        "a purple car in grass",
        ["the subject 'car'"],
        [("red", "blue")],
        notes=["The user confirmed the subject was a vehicle."],
    )
    assert out == (
        "a purple car in grass "
        "Note: the user may be wrong about: the subject 'car'. "
        "Do not penalise a photo for differing on those details. "
        "The user corrected 'red' to 'blue'. "
        "The user confirmed the subject was a vehicle."
    )


def test_demotion_guard_protects_union_top5():
    cands = [Candidate(f"img-{i:06d}", 0.90 - i * 0.01, ["original"],
                       {"original": 0.90 - i * 0.01}) for i in range(15)]
    scores = {f"img-{i:06d}.jpg": (3 if i == 0 else 10) for i in range(15)}
    paths = {c.image_id: f"/x/{c.image_id}.jpg" for c in cands}
    out = MuseReranker(FakeClaude(scores)).rerank(
        cands, paths, "a photo", top_n=15, weight=0.65)
    pos = [o.image_id for o in out].index("img-000000")
    assert pos < 10  # middling Muse score cannot bury a union top-5


def test_colour_match_soft_bonus():
    cands = [Candidate(c.image_id, 0.25, c.branches, {"original": 0.25})
             for c in _cands()]
    paths = {c.image_id: f"/x/{c.image_id}.jpg" for c in cands}
    r = MuseReranker(FakeClaude({"img-000000.jpg": 5, "img-000001.jpg": 5,
                                 "img-000002.jpg": 5}))
    out = r.rerank(cands, paths, "a photo", top_n=3, weight=0.45,
                   colour_matches={"img-000002": True,
                                   "img-000000": False,
                                   "img-000001": None})
    by_id = {o.image_id: o for o in out}
    assert by_id["img-000002"].final > by_id["img-000001"].final
    assert by_id["img-000001"].final > by_id["img-000000"].final


def test_rerank_failure_is_neutral_not_fatal():
    class Boom:
        def chat_with_image(self, path, prompt):
            raise RuntimeError("api down")

    paths = {c.image_id: f"/x/{c.image_id}.jpg" for c in _cands()}
    out = MuseReranker(Boom()).rerank(_cands(), paths, "a photo", top_n=3)
    assert [o.image_id for o in out] == ["img-000000", "img-000001", "img-000002"]
    assert all(o.muse is None for o in out)


def test_rerank_client_uses_stand_in_only_when_configured(monkeypatch):
    from backend.recall.providers import MUSE_MODEL, rerank_client

    monkeypatch.setenv("META_MODEL_API_KEY", "meta-key")
    monkeypatch.delenv("MUSE_MODEL", raising=False)
    monkeypatch.delenv("RERANK_BASE_URL", raising=False)
    assert rerank_client().model == MUSE_MODEL
    monkeypatch.setenv("RERANK_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("RERANK_API_KEY", "other-key")
    monkeypatch.setenv("RERANK_MODEL", "stand-in")
    c = rerank_client()
    assert (c.model, c.base_url, c.api_key) == (
        "stand-in", "https://example.test/v1", "other-key")


def test_muse_host_is_switched_by_env(monkeypatch):
    from backend.recall.providers import BASE_URL, MuseClient

    monkeypatch.setenv("META_MODEL_API_KEY", "meta-key")
    monkeypatch.delenv("MUSE_BASE_URL", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    meta = MuseClient()
    assert meta.base_url == BASE_URL
    assert meta._reasoning() == {"reasoning_effort": "low"}
    monkeypatch.setenv("MUSE_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("MUSE_API_KEY", "router-key")
    router = MuseClient()
    assert (router.base_url, router.api_key) == ("https://openrouter.ai/api/v1", "router-key")
    assert router._reasoning() == {"reasoning": {"effort": "low"}}


def test_shared_cache_makes_repeat_scores_free():
    calls = []

    class Client:
        model = "m"

        def chat_with_image(self, path, prompt, max_tokens=800):
            calls.append(path)
            return '{"score": 7}'

    cache: dict = {}
    r = MuseReranker(Client(), max_workers=1, cache=cache)
    assert r._score(("a.jpg", "d")) == 0.7
    assert r._score(("a.jpg", "d")) == 0.7
    assert calls == ["a.jpg"]
    r._score(("a.jpg", "other"))
    assert calls == ["a.jpg", "a.jpg"]


def test_hourly_limit_blocks_then_recovers(monkeypatch):
    from backend import app as a

    monkeypatch.setattr(a, "RERANKS_PER_HOUR", 2)
    monkeypatch.setattr(a, "_rerank_times", [])
    assert a._take_rerank_slot(1000.0) and a._take_rerank_slot(1001.0)
    assert not a._take_rerank_slot(1002.0)
    assert a._take_rerank_slot(1000.0 + 3601)


def _candidates():
    return [Candidate("a", 0.9), Candidate("b", 0.5), Candidate("c", 0.1)]


PATHS = {"a": "a.jpg", "b": "b.jpg", "c": "c.jpg"}


class _Fails:
    model = "main"

    def chat_with_image(self, *a, **k):
        raise RuntimeError("402")


def _setup(monkeypatch, primary, logits):
    from backend import app as a
    from backend.recall import providers as p

    monkeypatch.setattr(a, "rerank_client", lambda: primary)
    monkeypatch.setattr(a, "_primary_retry_at", [0.0])
    monkeypatch.setattr(a, "_rerank_times", [])
    monkeypatch.setattr(a, "_rerank_scores", {})
    monkeypatch.setenv("BACKUP_RERANK_API_KEY", "k")
    monkeypatch.setattr(p.BackupReranker, "logits", lambda self, d, ps: logits)
    return a


def test_ai_check_uses_backup_when_main_fails_and_waits_before_retry(monkeypatch):
    a = _setup(monkeypatch, _Fails(), {"a.jpg": -3.0, "b.jpg": 2.0, "c.jpg": -1.0})
    ranked, model = a._ai_check(_candidates(), PATHS, "d")
    assert model == "nvidia/nemotron-3.5-rerank-8b"
    assert ranked[0].image_id == "b"
    assert a._primary_retry_at[0] > 0
    assert len(a._rerank_times) == 1
    a._ai_check(_candidates(), PATHS, "d")
    assert len(a._rerank_times) == 1  # the main model is not tried again yet


def test_ai_check_keeps_main_model_when_it_works(monkeypatch):
    class Good:
        model = "main"

        def chat_with_image(self, *a, **k):
            return '{"score": 5}'

    a = _setup(monkeypatch, Good(), {})
    _ranked, model = a._ai_check(_candidates(), PATHS, "d")
    assert model == "main"


def test_ai_check_raises_when_nothing_can_score(monkeypatch):
    import pytest

    a = _setup(monkeypatch, _Fails(), {})
    with pytest.raises(RuntimeError):
        a._ai_check(_candidates(), PATHS, "d")


def test_backup_is_off_without_key(monkeypatch):
    import pytest

    from backend.recall.providers import BackupReranker, ProviderDisabled

    monkeypatch.delenv("BACKUP_RERANK_API_KEY", raising=False)
    with pytest.raises(ProviderDisabled):
        BackupReranker()
