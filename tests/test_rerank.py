from backend.recall.rank import Candidate
from backend.recall.rerank import MuseReranker


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


def test_rerank_failure_is_neutral_not_fatal():
    class Boom:
        def chat_with_image(self, path, prompt):
            raise RuntimeError("api down")

    paths = {c.image_id: f"/x/{c.image_id}.jpg" for c in _cands()}
    out = MuseReranker(Boom()).rerank(_cands(), paths, "a photo", top_n=3)
    assert [o.image_id for o in out] == ["img-000000", "img-000001", "img-000002"]
    assert all(o.muse is None for o in out)
