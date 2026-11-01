"""Question planner.

Priority (deterministic first, Muse when available):
 1. Clue-uncertainty: ask which detail the user might be
    misremembering. Grounded in our own mechanism: the answer
    directly re-weights the leave-one-out branch for that clue.
 2. Attribute question about the most-doubted clue (Muse Spark
    generates options from candidate context; template fallback).
 3. Uncertainty again for any remaining clue.

Every question offers "Not sure", which is always neutral.
Hard cap of MAX_QUESTIONS lives in the session, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NOT_SURE = "Not sure"
YES_SURE = "Yes, I am sure"
SOMETHING_ELSE = "It was something else"
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
           "pink", "brown", "black", "white", "gray"]


@dataclass
class Question:
    qid: str
    kind: str  # "subject" | "details" | "category" | "uncertainty" | "attribute"
    text: str
    options: list[str]
    clue_index: int | None = None
    asked: bool = field(default=False)


def subject_question(clue, idx: int, n: int) -> Question:
    return Question(
        qid=f"q{n}",
        kind="subject",
        text=f"Are you sure the main subject was '{clue.text}'?",
        options=[YES_SURE, NOT_SURE, SOMETHING_ELSE],
        clue_index=idx,
    )


def dimension_question(clue, idx: int, n: int) -> Question:
    """Ask only the bundled dimension (colour or count), not the
    whole subject, so a user sure of the noun can doubt the rest."""
    low = clue.text.lower()
    if clue.kind == "colour":
        colour = next((c for c in COLOURS if c in low), None)
        what = f"the {clue.referent or 'main object'} was {colour}" if colour \
            else f"'{clue.text}'"
        return Question(
            qid=f"q{n}", kind="dimension",
            text=f"Are you sure {what}?",
            options=[YES_SURE, NOT_SURE, SOMETHING_ELSE], clue_index=idx)
    return Question(
        qid=f"q{n}", kind="dimension",
        text=f"Are you sure it was '{clue.text}'?",
        options=[YES_SURE, NOT_SURE, SOMETHING_ELSE], clue_index=idx)


def replacement_question(clue, idx: int, n: int,
                         options: list[str] | None = None) -> Question:
    opts = list(options) if options else ["dog", "cat", "car", "bus",
                                          "bicycle", "boat", "train",
                                          "bird", "horse"]
    return Question(
        qid=f"q{n}", kind="attribute",
        text=f"What was the main subject instead of '{clue.text}'?",
        options=opts + [NOT_SURE], clue_index=idx,
    )


CATEGORY_OPTIONS = ["A vehicle", "An animal", "A person"]


def category_question(n: int) -> Question:
    return Question(
        qid=f"q{n}",
        kind="category",
        text="You weren't sure about the subject: do you remember "
             "what kind of thing it was?",
        options=["A vehicle", "An animal", "A person", NOT_SURE],
        clue_index=None,
    )


def details_question(n: int) -> Question:
    return Question(
        qid=f"q{n}",
        kind="details",
        text="Can you give one more detail: where it was, the colours, "
             "what was happening, or anything else you remember: so we "
             "can search without the subject?",
        options=[NOT_SURE],
        clue_index=None,
    )


def uncertainty_question(clues: list, idx: int, n: int) -> Question:
    return Question(
        qid=f"q{n}",
        kind="uncertainty",
        text="Which detail might you be misremembering? "
             "Pick the one you're least sure about.",
        options=[c.text for c in clues] + [NOT_SURE],
        clue_index=None,
    )


def attribute_template(clue, idx: int, n: int,
                       options: list[str] | None = None) -> Question:
    if options is not None:
        if clue.kind == "colour":
            text = f"What colour was the {clue.referent or 'main object'}?"
        elif clue.kind == "count":
            text = f"How many {clue.referent or 'were there'}?"
        elif clue.kind == "setting":
            text = "Where was this taken?"
        elif clue.kind in ("object", "person"):
            text = f"What was the {clue.referent or 'main subject'} instead?"
        else:
            text = f"About “{clue.text}”: which is closest?"
        seen: list[str] = []
        for o in options:
            if o not in seen and o != NOT_SURE:
                seen.append(o)
        return Question(
            qid=f"q{n}", kind="attribute", text=text,
            options=seen + [NOT_SURE], clue_index=idx,
        )
    if clue.kind == "colour":
        return Question(
            qid=f"q{n}", kind="attribute",
            text=f"What colour was the {clue.referent or 'main object'}?",
            options=COLOURS + [NOT_SURE], clue_index=idx,
        )
    if clue.kind == "count":
        return Question(
            qid=f"q{n}", kind="attribute",
            text=f"How many {clue.referent or 'were there'}?",
            options=["1", "2", "3", "4 or more", NOT_SURE], clue_index=idx,
        )
    if clue.kind == "setting":
        return Question(
            qid=f"q{n}", kind="attribute",
            text="Where was this taken?",
            options=["indoors", "outdoors (nature)", "outdoors (city)",
                     "at the beach / water", NOT_SURE], clue_index=idx,
        )
    if clue.kind in ("object", "person"):
        return Question(
            qid=f"q{n}", kind="attribute",
            text=f"What was the {clue.referent or 'main subject'} instead?",
            options=["dog", "cat", "car", "bus", "bicycle", "boat",
                     "train", "bird", "horse", NOT_SURE], clue_index=idx,
        )
    return Question(
        qid=f"q{n}", kind="attribute",
        text=f"About “{clue.text}”: which is closest?",
        options=[clue.text, NOT_SURE], clue_index=idx,
    )
