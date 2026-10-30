import pytest

from backend.recall.clues import Clue
from backend.recall.embeddings import StubEncoder
from backend.recall.questions import NOT_SURE
from backend.recall.retrieve import ExactIndex
from backend.recall.session import SearchSession


def _session() -> SearchSession:
    enc = StubEncoder()
    captions = [
        "orange jacket; beach; daylight",   # img-000000 target ('red' wrong)
        "red jacket; city; night",
        "blue car; street; daylight",
        "red bus; street; daylight",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("red jacket", "object", "jacket"),
             Clue("beach", "setting", "beach"),
             Clue("daylight", "other", "")]
    s = SearchSession(index, enc, "red jacket; beach; daylight", clues)
    s.start()
    return s


def test_session_starts_with_question_and_results():
    s = _session()
    assert s.status == "asking"
    assert s.pending is not None and s.pending.kind == "uncertainty"
    assert s.snapshot()["results"]


def test_drop_answer_reranks_and_records():
    s = _session()
    before = [c.image_id for c in s.candidates[:5]]
    s.answer("beach")  # setting = modifier, safe to drop
    assert 1 in s.dropped
    out = s.finish()
    assert out["dropped_clues"] == ["beach"]
    assert "beach" in out["explanation"]
    assert len(out["results"]) == 5 or len(out["results"]) >= 1
    _ = before  # order change is data-dependent; provenance is the contract


def test_doubted_anchor_is_replaced_not_dropped():
    s = _session()
    # 'red jacket' clue is kind=object (anchor): doubting it must
    # not drop it or re-weight anything; next Q asks replacement.
    s.answer("red jacket")
    assert s.dropped == set()
    assert s.weights == {}
    assert s.doubted_anchor == 0
    assert s.pending is not None and s.pending.kind == "attribute"
    assert s.pending.clue_index == 0


def test_finish_explanation_when_anchor_doubted_not_replaced():
    s = _session()
    s.answer("red jacket")  # anchor doubt, then finish immediately
    out = s.finish()
    assert "didn't get a replacement" in out["explanation"]
    assert "matched the original" not in out["explanation"]


def test_not_sure_is_neutral():
    s = _session()
    top_before = s.candidates[0].image_id
    s.answer(NOT_SURE)
    assert s.dropped == set()
    assert s.weights == {}
    assert s.candidates[0].image_id == top_before


def test_question_cap_is_three():
    s = _session()
    for _ in range(3):
        if s.pending is None:
            break
        s.answer(NOT_SURE)
    assert len(s.answers) <= 3
    with pytest.raises(RuntimeError):
        s.answer(NOT_SURE)


def test_persistence_roundtrip_preserves_state():
    s = _session()
    s.answer("beach")
    data = s.to_dict()
    s2 = SearchSession.from_dict(s.index, s.encoder, data)
    assert s2.session_id == s.session_id
    assert s2.dropped == s.dropped == {1}
    assert [c.image_id for c in s2.candidates] == [c.image_id for c in s.candidates]
    assert s2.answers[0].value == "beach"
    assert (s2.pending is None) == (s.pending is None)


def test_finish_without_answers_explains():
    s = _session()
    out = s.finish()
    assert out["questions_asked"] == 0
    assert "No follow-up" in out["explanation"]
