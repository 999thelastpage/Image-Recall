from backend.recall.clues import Clue, dedupe_clues, hypotheses, parse_fallback


def test_dedupe_drops_contained_and_duplicate_chips():
    clues = [Clue("train", "object"), Clue("mini orange train", "object"),
             Clue("Train", "object"), Clue("children", "person"),
             Clue("riding on mini orange train", "activity")]
    out = dedupe_clues(clues)
    texts = [c.text for c in out]
    assert "train" not in texts and "Train" not in texts
    assert "mini orange train" in texts and "children" in texts


def test_fallback_splits_on_semicolons():
    clues = parse_fallback("red jacket; at the beach; two people")
    assert [c.text for c in clues] == ["red jacket", "at the beach", "two people"]
    assert [c.kind for c in clues] == ["colour", "setting", "count"]


def test_fallback_single_clue():
    # Multi-fact sentence now splits; truly single-fact stays one clue.
    assert parse_fallback("a dog") == [Clue("a dog")]
    assert parse_fallback("   ") == []
    park = parse_fallback("a dog in a park")
    assert len(park) >= 2
    assert {c.kind for c in park} >= {"object", "setting"}


def test_fallback_plain_sentence_never_collapses():
    for query in [
        "a big purple bus parked in a parking spot",
        "a man riding a bicycle next to a red car",
        "the cake on the kitchen table",
        "two dogs running through the snow in the mountains",
    ]:
        clues = parse_fallback(query)
        assert len(clues) >= 2, (query, clues)


def test_hypotheses_original_plus_leave_one_out():
    clues = [Clue("red jacket"), Clue("beach"), Clue("daylight")]
    hyps = hypotheses(clues)
    assert [h.label for h in hyps] == ["original", "drop:0", "drop:1", "drop:2"]
    assert hyps[1].query_text == "beach; daylight"
    assert hyps[1].dropped_index == 0


def test_hypotheses_relax_colour_keeps_noun():
    clues = [Clue("purple bus", "colour", "bus"), Clue("in grass", "setting", "grass")]
    labels = [h.label for h in hypotheses(clues)]
    assert "relax:0" in labels
    relax = next(h for h in hypotheses(clues) if h.label == "relax:0")
    assert relax.query_text == "bus; in grass"


def test_hypotheses_substitution_branches():
    clues = [Clue("purple bus", "colour", "bus"), Clue("in grass", "setting", "grass")]
    hyps = hypotheses(clues, substitution_colours=("red", "blue"), max_substitutions=2)
    subs = [h for h in hyps if h.label.startswith("sub:")]
    assert {h.query_text for h in subs} == {"red bus; in grass", "blue bus; in grass"}


def test_hypotheses_single_clue_has_no_drops():
    hyps = hypotheses([Clue("only clue")])
    assert len(hyps) == 1 and hyps[0].dropped_index is None


def test_fallback_splits_on_commas_too():
    clues = parse_fallback("a red jacket, at the beach, and two dogs")
    assert [c.text for c in clues] == ["a red jacket", "at the beach", "two dogs"]
