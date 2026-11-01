import re

import pytest

from backend.recall.clues import Clue
from backend.recall.embeddings import StubEncoder
from backend.recall.questions import NOT_SURE, SOMETHING_ELSE, YES_SURE
from backend.recall.retrieve import ExactIndex
from backend.recall.session import SearchSession, _subjectless_score


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


def _subject_only_session() -> SearchSession:
    enc = StubEncoder()
    captions = [
        "orange jacket; beach; daylight",
        "red jacket; city; night",
        "blue car; street; daylight",
        "red bus; street; daylight",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("red jacket", "object", "jacket")]
    s = SearchSession(index, enc, "red jacket", clues)
    s.start()
    return s


def test_session_starts_with_question_and_results():
    s = _session()
    assert s.status == "asking"
    assert s.pending is not None and s.pending.kind == "subject"
    assert s.snapshot()["results"]


def test_subject_question_is_first_and_names_anchor():
    s = _session()
    assert s.pending is not None
    assert s.pending.kind == "subject"
    assert "red jacket" in s.pending.text
    assert s.pending.clue_index == 0


def test_confirming_subject_changes_no_weights_or_dropped():
    s = _session()
    s.answer(YES_SURE)
    assert s.dropped == set()
    assert s.weights == {}
    assert s.subject_checked is True
    assert s.pending is not None and s.pending.kind == "uncertainty"


def test_not_sure_on_subject_with_other_clues_drops_anchor_fuzzy():
    s = _session()
    before = {c.image_id: c.score for c in s.candidates}
    s.answer(NOT_SURE)
    assert s.dropped == {0}
    assert s.subject_checked is True
    # no replacement question next
    assert s.pending is None or s.pending.kind != "attribute"
    after = {c.image_id: c for c in s.candidates}
    for iid, old in before.items():
        assert "revised" in after[iid].branch_scores
        assert after[iid].score >= old - 1e-9


def test_subject_only_not_sure_asks_details_then_free_text_searches():
    s = _subject_only_session()
    assert s.pending is not None and s.pending.kind == "subject"
    s.answer(NOT_SURE)
    assert s.pending is not None and s.pending.kind == "details"
    assert s.dropped == {0}
    s.answer("at the beach at sunset")
    assert s.clues[-1].text == "at the beach at sunset"
    assert s.clues[-1].kind == "other"
    assert "at the beach at sunset" in s.description
    assert s.dropped == {0}
    assert s.candidates
    for c in s.candidates:
        assert "revised" in c.branch_scores


def test_drop_answer_reranks_and_records():
    s = _session()
    before = [c.image_id for c in s.candidates[:5]]
    s.answer(YES_SURE)
    s.answer("beach")  # setting = modifier, safe to drop
    assert 1 in s.dropped
    out = s.finish()
    assert out["dropped_clues"] == ["beach"]
    assert "beach" in out["explanation"]
    assert len(out["results"]) == 5 or len(out["results"]) >= 1
    _ = before  # order change is data-dependent; provenance is the contract


def test_doubted_anchor_is_replaced_not_dropped():
    s = _session()
    # 'It was something else' on the subject question: doubting the
    # anchor must not drop it or re-weight anything; next Q asks
    # replacement.
    s.answer(SOMETHING_ELSE)
    assert s.dropped == set()
    assert s.weights == {}
    assert s.doubted_anchor == 0
    assert s.pending is not None and s.pending.kind == "attribute"
    assert s.pending.clue_index == 0


def test_finish_explanation_shows_correction_as_from_to():
    s = _session()
    s.answer(SOMETHING_ELSE)        # doubt the anchor
    s.answer("bus")                 # replace it (noun attribute answer)
    out = s.finish()
    # subject substitution preserves the modifier: red jacket -> red bus
    assert "'red jacket' → 'red bus'" in out["explanation"]


def test_finish_explanation_when_anchor_doubted_not_replaced():
    s = _session()
    s.answer(SOMETHING_ELSE)  # anchor doubt, then finish immediately
    out = s.finish()
    assert "didn't get a replacement" in out["explanation"]
    assert "matched the original" not in out["explanation"]


def test_answer_rescores_every_candidate_and_never_lowers_scores():
    s = _session()
    before = {c.image_id: c.score for c in s.candidates}
    s.answer(YES_SURE)
    s.answer("beach")  # drop a modifier -> revised query re-entry
    after = {c.image_id: c for c in s.candidates}
    for iid, old in before.items():
        # every pre-existing candidate was rescored under 'revised'
        assert "revised" in after[iid].branch_scores
        # and its final score did not fall
        assert after[iid].score >= old - 1e-9


def test_not_sure_is_neutral():
    s = _session()
    s.answer(YES_SURE)
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
    s.answer(YES_SURE)
    s.answer("beach")
    data = s.to_dict()
    s2 = SearchSession.from_dict(s.index, s.encoder, data)
    assert s2.session_id == s.session_id
    assert s2.dropped == s.dropped == {1}
    assert [c.image_id for c in s2.candidates] == [c.image_id for c in s.candidates]
    assert s2.answers[1].value == "beach"
    assert (s2.pending is None) == (s.pending is None)
    assert s2.subject_checked is True


def test_persistence_roundtrip_preserves_subject_checked():
    s = _session()
    s.answer(YES_SURE)
    data = s.to_dict()
    assert data["subject_checked"] is True
    s2 = SearchSession.from_dict(s.index, s.encoder, data)
    assert s2.subject_checked is True
    assert s2.needs_details is False


def test_old_format_dict_still_loads():
    s = _session()
    s.answer(YES_SURE)
    data = s.to_dict()
    data.pop("subject_checked", None)
    data.pop("needs_details", None)
    s2 = SearchSession.from_dict(s.index, s.encoder, data)
    # default derived from the existing 'subject' answer
    assert s2.subject_checked is True
    assert s2.needs_details is False


def test_finish_without_answers_explains():
    s = _session()
    out = s.finish()
    assert out["questions_asked"] == 0
    assert "No follow-up" in out["explanation"]


class _RecordingEncoder(StubEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def encode_texts(self, texts):
        self.seen.extend(texts)
        return super().encode_texts(texts)


def _purple_session(enc=None) -> SearchSession:
    enc = enc or StubEncoder()
    captions = [
        "purple bus parked in grass",
        "purple car parked on street",
        "red car parked in grass",
        "blue bus parked in grass",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("purple car", "colour", "car"),
             Clue("car parked", "activity", "car"),
             Clue("in grass", "setting", "grass")]
    s = SearchSession(index, enc, "a purple car parked in grass", clues)
    s.start()
    return s


def test_colour_kind_subject_bearing_clue_gets_subject_question_first():
    s = _purple_session()
    assert s.pending is not None
    assert s.pending.kind == "subject"
    assert "purple car" in s.pending.text
    assert s.pending.clue_index == 0


def test_something_else_on_colour_anchor_asks_replacement_nouns():
    s = _purple_session()
    s.answer(SOMETHING_ELSE)
    assert s.dropped == set()
    assert s.doubted_anchor == 0
    assert s.pending is not None and s.pending.kind == "dimension"
    s.answer(YES_SURE)
    assert s.pending is not None and s.pending.kind == "attribute"
    assert s.pending.clue_index == 0
    assert "bus" in s.pending.options
    assert "purple" not in s.pending.options
    assert "red" not in s.pending.options


def test_bus_answer_preserves_colour_and_swaps_subject_everywhere():
    enc = _RecordingEncoder()
    s = _purple_session(enc)
    s.answer(SOMETHING_ELSE)
    s.answer(YES_SURE)  # colour settled, noun is the doubt
    before = len(enc.seen)
    s.answer("bus")
    assert s.answers[-1].value == "purple bus"
    answer_queries = [t for t in enc.seen[before:] if "purple bus" in t]
    assert answer_queries, enc.seen
    assert any("bus parked" in t for t in answer_queries)
    assert not any(re.search(r"\bcar\b", t) for t in answer_queries)
    out = s.finish()
    assert "'purple car' → 'purple bus'" in out["explanation"]


def test_not_sure_on_colour_anchor_revised_query_drops_car_keeps_purple():
    enc = _RecordingEncoder()
    s = _purple_session(enc)
    before = len(enc.seen)
    s.answer(NOT_SURE)
    assert s.dropped == {0}
    revised = enc.seen[before:]
    assert revised, enc.seen
    q = revised[-1]
    assert not re.search(r"\bcar\b", q.lower()), q
    assert "purple" in q.lower(), q
    by_id = {c.image_id: c for c in s.candidates}
    purple_bus = by_id["img-000000"]  # "purple bus parked in grass"
    assert "revised" in purple_bus.branch_scores
    assert "revised" in purple_bus.branches


def test_uncertainty_after_subject_confirmation_excludes_anchor():
    s = _purple_session()
    s.answer(YES_SURE)
    # bundled colour dimension is asked on its own first
    assert s.pending is not None and s.pending.kind == "dimension"
    s.answer(YES_SURE)
    assert s.pending is not None and s.pending.kind == "uncertainty"
    assert "purple car" not in s.pending.options
    assert "car parked" in s.pending.options
    assert "in grass" in s.pending.options


def test_subject_not_sure_with_other_clues_asks_category_next():
    s = _purple_session()
    assert s.pending is not None and s.pending.kind == "subject"
    s.answer(NOT_SURE)
    assert s.pending is not None and s.pending.kind == "category"
    assert s.pending.options == ["A vehicle", "An animal", "A person", NOT_SURE]
    assert s.pending.text == (
        "You weren't sure about the subject: do you remember "
        "what kind of thing it was?"
    )
    assert s.needs_category is True


def test_category_vehicle_runs_subjectless_category_search():
    enc = _RecordingEncoder()
    s = _purple_session(enc)
    s.answer(NOT_SURE)
    assert s.pending is not None and s.pending.kind == "category"
    before = {c.image_id: c.score for c in s.candidates}
    seen_before = len(enc.seen)
    s.answer("A vehicle")
    assert s.needs_category is False
    # category branch on candidates, scores never lowered
    for iid, old in before.items():
        cand = next(c for c in s.candidates if c.image_id == iid)
        assert "category" in cand.branch_scores
        assert "category" in cand.branches
        assert cand.score >= old - 1e-9
    # recorded category query: subject-less (no 'car' token), has category word
    new_queries = enc.seen[seen_before:]
    assert new_queries, enc.seen
    cat_q = new_queries[-1]
    assert not re.search(r"\bcar\b", cat_q.lower()), cat_q
    assert "vehicle" in cat_q.lower(), cat_q
    assert "purple" in cat_q.lower(), cat_q


def test_category_not_sure_is_neutral():
    s = _purple_session()
    s.answer(NOT_SURE)
    assert s.pending is not None and s.pending.kind == "category"
    before_scores = {c.image_id: c.score for c in s.candidates}
    s.answer(NOT_SURE)
    assert s.needs_category is False
    assert "category" not in s.weights
    for c in s.candidates:
        assert "category" not in c.branch_scores
        assert "category" not in c.branches
        assert c.score >= before_scores[c.image_id] - 1e-9


def test_subject_only_asks_details_before_category():
    s = _subject_only_session()
    s.answer(NOT_SURE)
    # details still comes first even though category is also owed
    assert s.pending is not None and s.pending.kind == "details"
    assert s.needs_category is True
    s.answer("at the beach at sunset")
    assert s.pending is not None and s.pending.kind == "category"


def test_yes_sure_and_something_else_never_ask_category():
    s = _session()
    s.answer(YES_SURE)
    assert s.needs_category is False
    assert s.pending is None or s.pending.kind != "category"

    s2 = _session()
    s2.answer(SOMETHING_ELSE)
    assert s2.needs_category is False
    assert s2.pending is not None and s2.pending.kind == "attribute"


def test_persistence_roundtrip_preserves_needs_category():
    s = _purple_session()
    s.answer(NOT_SURE)
    assert s.needs_category is True
    data = s.to_dict()
    assert data["needs_category"] is True
    s2 = SearchSession.from_dict(s.index, s.encoder, data)
    assert s2.needs_category is True
    assert s2.pending is not None and s2.pending.kind == "category"
    # after answering category, round-trip keeps it cleared
    s.answer("A vehicle")
    data2 = s.to_dict()
    assert data2["needs_category"] is False
    s3 = SearchSession.from_dict(s.index, s.encoder, data2)
    assert s3.needs_category is False


def test_old_format_dict_derives_needs_category():
    s = _purple_session()
    s.answer(NOT_SURE)
    data = s.to_dict()
    data.pop("needs_category", None)
    s2 = SearchSession.from_dict(s.index, s.encoder, data)
    # subject answered NOT_SURE, no category answer yet -> owed
    assert s2.needs_category is True


def _dogs_session(enc=None, clues=None) -> SearchSession:
    enc = enc or StubEncoder()
    captions = [
        "two cats together on a sofa",
        "two dogs together on a sofa",
        "one dog alone in a field",
        "two birds together in a tree",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    if clues is None:
        clues = [Clue("two dogs", "count", "dogs"),
                 Clue("dogs", "object", "dogs"),
                 Clue("together", "other", "dogs")]
    s = SearchSession(index, enc, "two dogs together", clues)
    s.start()
    return s


def test_two_dogs_subject_question_names_count_chip_first():
    s = _dogs_session()
    assert s.pending is not None
    assert s.pending.kind == "subject"
    assert "two dogs" in s.pending.text
    assert s.pending.clue_index == 0


def test_two_dogs_something_else_replacement_and_correction():
    enc = _RecordingEncoder()
    s = _dogs_session(enc)
    assert s.pending is not None and s.pending.kind == "subject"
    assert s.pending.clue_index == 0
    s.answer(SOMETHING_ELSE)
    assert s.doubted_anchor == 0
    assert s.pending is not None and s.pending.kind == "dimension"
    assert "two dogs" in s.pending.text
    s.answer(YES_SURE)
    assert s.pending is not None and s.pending.kind == "attribute"
    before = len(enc.seen)
    s.answer("cat")
    assert s.answers[-1].value == "two cats"
    answer_queries = enc.seen[before:]
    assert answer_queries, enc.seen
    assert any("two cats" in t for t in answer_queries), answer_queries
    assert not any(re.search(r"\bdogs?\b", t) for t in answer_queries), answer_queries
    # answer branch is recorded on candidates
    assert any("answer:0" in c.branches for c in s.candidates)
    out = s.finish()
    assert "'two dogs' → 'two cats'" in out["explanation"]


def test_two_dogs_singular_referent_still_swaps_to_two_cats():
    enc = _RecordingEncoder()
    clues = [Clue("two dogs", "count", "dog"),
             Clue("dogs", "object", "dogs"),
             Clue("together", "other", "dogs")]
    s = _dogs_session(enc, clues=clues)
    assert s.pending is not None and s.pending.kind == "subject"
    assert s.pending.clue_index == 0
    assert "two dogs" in s.pending.text
    s.answer(SOMETHING_ELSE)
    assert s.pending is not None and "two dogs" in s.pending.text
    s.answer(YES_SURE)  # count settled, noun is the doubt
    before = len(enc.seen)
    s.answer("cat")
    assert s.answers[-1].value == "two cats"
    answer_queries = enc.seen[before:]
    assert any("two cats" in t for t in answer_queries), answer_queries
    assert not any(re.search(r"\bdogs?\b", t) for t in answer_queries), answer_queries
    out = s.finish()
    assert "'two dogs' → 'two cats'" in out["explanation"]


def _dogs_together_session() -> SearchSession:
    enc = StubEncoder()
    captions = [
        "two dogs; together",   # img-000000: exact original-query match
        "two cats; together",   # img-000001
        "one dog; alone",       # img-000002
        "two birds; together",  # img-000003
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("two dogs", "count", "dogs"),
             Clue("together", "other", "dogs")]
    s = SearchSession(index, enc, "two dogs together", clues)
    s.start()
    return s


def test_subject_not_sure_reorders_by_subjectless_score():
    s = _dogs_together_session()
    # premise: the dog photo leads under the original query
    assert s.candidates[0].image_id == "img-000000"
    count_before = len(s.candidates)
    s.answer(NOT_SURE)
    assert s.dropped == {0}
    # nothing removed, only re-ordered
    assert len(s.candidates) == count_before
    # live order follows subject-less branch evidence
    expected = sorted(
        s.candidates,
        key=lambda c: (-_subjectless_score(c), -c.score, c.image_id),
    )
    assert [c.image_id for c in s.candidates] == \
        [c.image_id for c in expected]
    # the candidate that only led via the original query no longer leads
    assert s.candidates[0].image_id != "img-000000"
    # snapshot (live order) follows the same contract
    snap_ids = [r["image_id"] for r in s.snapshot()["results"]]
    assert snap_ids == [c.image_id for c in expected][:len(snap_ids)]
    # finish() top results are ordered by subject-less evidence too
    by_id = {c.image_id: c for c in s.candidates}
    out = s.finish()
    top_scores = [_subjectless_score(by_id[r["image_id"]])
                  for r in out["results"]]
    assert top_scores == sorted(top_scores, reverse=True)
    assert [r["image_id"] for r in out["results"]] == \
        [c.image_id for c in expected][:len(out["results"])]
    assert out["results"][0]["image_id"] != "img-000000"


def test_subjectless_score_defaults_to_candidate_score():
    from backend.recall.rank import Candidate

    c = Candidate("img-x", 0.42, ["original"], {"original": 0.42})
    assert _subjectless_score(c) == 0.42
    c2 = Candidate("img-y", 0.9, ["original", "revised"],
                   {"original": 0.9, "revised": 0.3})
    assert _subjectless_score(c2) == 0.3


def test_activity_other_clues_never_get_fallback_attribute_question():
    enc = StubEncoder()
    captions = [
        "orange jacket; beach; daylight",
        "red jacket; city; night",
        "blue car; street; daylight",
        "red bus; street; daylight",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("red jacket", "object", "jacket"),
             Clue("dancing", "activity", ""),
             Clue("daylight", "other", "")]
    s = SearchSession(index, enc, "red jacket; dancing; daylight", clues)
    s.start()
    assert s.pending is not None and s.pending.kind == "subject"
    s.answer(YES_SURE)
    assert s.pending is not None and s.pending.kind == "uncertainty"
    s.answer(NOT_SURE)
    # attribute loop must skip activity/other: no vague fallback question
    assert s.pending is None or s.pending.kind != "attribute"
    if s.pending is not None:
        assert "which is closest" not in s.pending.text


def test_confident_colour_path_asks_dimension_then_uncertainty():
    s = _purple_session()
    kinds = [s.pending.kind if s.pending else None]
    s.answer(YES_SURE)
    kinds.append(s.pending.kind if s.pending else None)
    assert s.pending is not None and "purple" in s.pending.text
    s.answer(YES_SURE)
    kinds.append(s.pending.kind if s.pending else None)
    assert kinds == ["subject", "dimension", "uncertainty"]


def test_subject_doubt_on_bundled_colour_settles_dimension_first():
    s = _purple_session()
    s.answer(SOMETHING_ELSE)
    # dimension is settled before the replacement noun is asked
    assert s.pending is not None and s.pending.kind == "dimension"
    s.answer(SOMETHING_ELSE)
    assert s.pending is not None and s.pending.kind == "attribute"
    assert "colour" in s.pending.text


def test_dimension_doubt_reaches_colour_attribute():
    s = _purple_session()
    s.answer(YES_SURE)
    assert s.pending is not None and s.pending.kind == "dimension"
    s.answer(SOMETHING_ELSE)
    assert s.pending is not None and s.pending.kind == "attribute"
    assert "colour" in s.pending.text
    assert "blue" in s.pending.options and "brown" in s.pending.options
    s.answer("blue")
    assert any("answer:0" in c.branches for c in s.candidates)


def test_invalid_answer_rejected_without_consuming_budget():
    import pytest

    s = _session()
    before = len(s.answers)
    with pytest.raises(RuntimeError):
        s.answer("blue")
    assert len(s.answers) == before
    assert s.pending is not None and s.pending.kind == "subject"


def test_uncertainty_never_offers_dropped_clue():
    s = _purple_session()
    s.answer(YES_SURE)
    s.answer(YES_SURE)  # dimension confirmed
    assert s.pending is not None and s.pending.kind == "uncertainty"
    s.answer("in grass")
    assert 2 in s.dropped
    if s.pending is not None and s.pending.kind == "uncertainty":
        assert "in grass" not in s.pending.options


def test_cake_setting_question_offers_kitchen_option():
    enc = StubEncoder()
    captions = ["cake kitchen table candles", "cake beach picnic",
                "dog park grass", "cat sofa indoors"]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("on the kitchen table", "setting", "table"),
             Clue("cake", "object", "cake")]
    s = SearchSession(index, enc, "the cake on the kitchen table", clues)
    s.start()
    s.answer(YES_SURE)
    s.answer(NOT_SURE)  # uncertainty, neutral
    assert s.pending is not None and s.pending.kind == "attribute"
    assert any("kitchen" in o for o in s.pending.options)


def test_colour_fix_after_subject_doubt_keeps_noun_and_drops_wrong_colour():
    enc = _RecordingEncoder()
    s = _purple_session(enc)
    s.answer(SOMETHING_ELSE)  # subject doubted
    s.answer(SOMETHING_ELSE)  # the colour is the wrong part
    assert s.pending is not None and s.pending.kind == "attribute"
    assert "colour" in s.pending.text
    before = len(enc.seen)
    s.answer("blue")
    answer_queries = enc.seen[before:]
    assert any("blue car" in t for t in answer_queries), answer_queries
    assert not any("purple" in t for t in answer_queries), answer_queries
    assert s.answers[-1].value == "blue"
    # the noun was never doubted once the colour took the blame
    assert s.pending is None or s.pending.kind != "attribute"
