"""Question planner.

Priority (deterministic first, Muse when available):
 1. Clue-uncertainty: ask which detail the user might be
    misremembering. Grounded in our own mechanism — the answer
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
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
           "pink", "brown", "black", "white", "gray"]


@dataclass
class Question:
    qid: str
    kind: str  # "uncertainty" | "attribute"
    text: str
    options: list[str]
    clue_index: int | None = None
    asked: bool = field(default=False)


def uncertainty_question(clues: list, idx: int, n: int) -> Question:
    return Question(
        qid=f"q{n}",
        kind="uncertainty",
        text="Which detail might you be misremembering? "
             "Pick the one you're least sure about.",
        options=[c.text for c in clues] + [NOT_SURE],
        clue_index=None,
    )


def attribute_template(clue, idx: int, n: int) -> Question:
    if clue.kind == "colour":
        return Question(
            qid=f"q{n}", kind="attribute",
            text=f"What colour was the {clue.referent or 'main object'}?",
            options=COLOURS[:6] + [NOT_SURE], clue_index=idx,
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
            options=["indoors", "outdoors — nature", "outdoors — city",
                     "at the beach / water", NOT_SURE], clue_index=idx,
        )
    return Question(
        qid=f"q{n}", kind="attribute",
        text=f"About “{clue.text}” — which is closest?",
        options=[clue.text, NOT_SURE], clue_index=idx,
    )
