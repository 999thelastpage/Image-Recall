from backend.recall.clues import Clue, dedupe_clues, hypotheses, parse_fallback


def test_dedupe_drops_contained_and_duplicate_chips():
    clues = [Clue("train"), Clue("mini orange train"), Clue("Train"),
             Clue("children"), Clue("riding on mini orange train")]
    out = dedupe_clues(clues)
    texts = [c.text for c in out]
    assert "train" not in texts and "Train" not in texts
    assert "mini orange train" in texts and "children" in texts


def test_fallback_splits_on_semicolons():
    clues = parse_fallback("red jacket; at the beach; two people")
    assert clues == [Clue("red jacket"), Clue("at the beach"), Clue("two people")]


def test_fallback_single_clue():
    assert parse_fallback("a dog in a park") == [Clue("a dog in a park")]
    assert parse_fallback("   ") == []


def test_hypotheses_original_plus_leave_one_out():
    clues = parse_fallback("red jacket; beach; daylight")
    hyps = hypotheses(clues)
    assert [h.label for h in hyps] == ["original", "drop:0", "drop:1", "drop:2"]
    assert hyps[1].query_text == "beach; daylight"
    assert hyps[1].dropped_index == 0


def test_hypotheses_single_clue_has_no_drops():
    hyps = hypotheses([Clue("only clue")])
    assert len(hyps) == 1 and hyps[0].dropped_index is None
