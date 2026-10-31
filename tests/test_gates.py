from backend.recall.embeddings import build_encoder
from backend.recall.rerank import _znorm
from scripts.encoder_ab import summarise
from scripts.g2_eval import classify_failure


def test_classify_failure_layers():
    assert classify_failure(100, None, None) == "coverage"
    assert classify_failure(100, 100, 100) == "pool-depth"
    assert classify_failure(100, 10, 20) == "ranking"
    assert classify_failure(2, 10, 20) == "ranking-demotion"
    assert classify_failure(100, 10, 3) == "hit"


def test_summarise_reports_ceiling_separately():
    out = summarise([1, 6, 60, None])
    assert out["recall@5"] == 0.25
    assert out["recall@50"] == 0.5
    assert out["median_rank"] == 6


def test_znorm_clips_outlier():
    out = _znorm([0.20, 0.21, 0.22, 0.90])
    assert out[-1] > 0.9
    assert 0.0 <= out[0] < out[1] < out[2] < 1.0
    assert _znorm([0.5, 0.5]) == [0.5, 0.5]


def test_pe_core_fails_loudly():
    try:
        build_encoder("facebook/PE-Core-L14-336")
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as e:
        assert "dedicated loader" in str(e)
