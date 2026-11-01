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
