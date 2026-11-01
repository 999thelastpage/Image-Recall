"""Clue parsing (fallback) and leave-one-clue-out hypotheses.

The primary parser is Muse Spark (see providers). This structured
fallback always works offline. Punctuated input is split and each
part is typed; a plain sentence is scanned with small lexicons for
colour, count, object, setting and activity so a multi-fact query
never collapses to a single clue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RAW_LABEL = "raw"

COLOUR_WORDS = ["red", "orange", "yellow", "green", "blue", "purple",
                "pink", "brown", "black", "white", "gray", "grey", "gold"]
COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
               "ten": 10, "couple": 2, "pair": 2}
OBJECT_NOUNS = {"dog", "dogs", "cat", "cats", "car", "cars", "bus", "buses",
                "bicycle", "bicycles", "bike", "boat", "train", "bird",
                "birds", "horse", "horses", "cake", "chair", "table",
                "jacket", "dress", "person", "people", "man", "woman",
                "boy", "girl", "child", "children"}
PERSON_NOUNS = {"person", "people", "man", "woman", "boy", "girl",
                "child", "children", "men", "women"}
SETTING_PHRASES = ["at the beach", "on a kitchen table", "on the kitchen table",
                   "in the mountains", "in the snow", "on a city street",
                   "on the street", "in a park", "in the park", "in grass",
                   "in the grass", "at night", "in a parking", "parking spot",
                   "parking lot", "kitchen table", "the beach", "the park",
                   "beach", "park", "kitchen", "street", "snow", "mountains",
                   "city", "indoors", "outdoors", "outside", "grass", "water"]
ACTIVITY_WORDS = {"riding", "running", "sitting", "standing", "eating",
                  "holding", "parked", "playing", "walking", "sleeping",
                  "jumping", "swimming", "driving", "cooking", "reading"}


@dataclass(frozen=True)
class Clue:
    text: str
    kind: str = "other"
    referent: str = ""


@dataclass(frozen=True)
class Hypothesis:
    label: str
    query_text: str
    dropped_index: int | None


def _colour_kind(c: Clue) -> Clue:
    """'orange raft' is a colour fact, whatever the parser called it, so
    the user can correct the colour and keep the noun."""
    words = re.findall(r"[a-z]+", c.text.lower())
    if c.kind in ("object", "other") and len(words) == 2 and words[0] in COLOUR_WORDS:
        return Clue(c.text, "colour", c.referent.strip() or words[1])
    return c


def dedupe_clues(clues: list[Clue]) -> list[Clue]:
    seen: set[str] = set()
    uniq: list[Clue] = []
    for c in map(_colour_kind, clues):
        key = c.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    return [
        c for c in uniq
        if not any(
            c is not other
            and (c.kind == other.kind or "other" in (c.kind, other.kind))
            and c.text.strip().lower() in other.text.strip().lower()
            and len(other.text) > len(c.text)
            for other in uniq
        )
    ]


def _referent(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    for w in reversed(words):
        if w in OBJECT_NOUNS:
            return w
    return words[-1] if words else ""


def _classify_part(part: str) -> Clue:
    low = part.lower()
    words = re.findall(r"[a-z]+|\d+", low)
    colour = next((w for w in words if w in COLOUR_WORDS), None)
    if colour:
        return Clue(part, "colour", _referent(part))
    if words and (words[0] in COUNT_WORDS or words[0].isdigit()):
        if any(w in OBJECT_NOUNS for w in words):
            return Clue(part, "count", _referent(part))
    if any(p in low for p in SETTING_PHRASES):
        return Clue(part, "setting", _referent(part))
    if any(w in PERSON_NOUNS for w in words):
        return Clue(part, "person", _referent(part))
    if any(w in OBJECT_NOUNS for w in words):
        return Clue(part, "object", _referent(part))
    if any(w in ACTIVITY_WORDS for w in words):
        return Clue(part, "activity", _referent(part))
    return Clue(part)


def _parse_sentence(description: str) -> list[Clue]:
    text = description.strip()
    low = " " + text.lower() + " "
    clues: list[Clue] = []
    covered: list[str] = []
    colour_m = None
    for colour in COLOUR_WORDS:
        colour_m = re.search(rf"\b{colour}\s+([a-z]+)", low)
        if colour_m:
            noun = colour_m.group(1)
            clues.append(Clue(f"{colour} {noun}", "colour", noun))
            covered.append(f"{colour} {noun}")
            break
    count_m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+)", low)
    if count_m and not any(count_m.group(0) in c for c in covered):
        clues.append(Clue(count_m.group(0), "count", count_m.group(2)))
        covered.append(count_m.group(0))
    for phrase in sorted(SETTING_PHRASES, key=len, reverse=True):
        if f" {phrase}" in low or f"{phrase} " in low:
            if not any(phrase in c for c in covered):
                clues.append(Clue(phrase, "setting", _referent(phrase)))
                covered.append(phrase)
            break
    act_m = next((w for w in re.findall(r"[a-z]+", low) if w in ACTIVITY_WORDS), None)
    if act_m and not any(act_m in c for c in covered):
        idx = low.find(act_m)
        frag = text[max(0, idx - 1): idx + 24].strip(" ,.;")
        frag = " ".join(frag.split()[:4])
        if len(frag) > 2:
            clues.append(Clue(frag, "activity", _referent(frag)))
            covered.append(act_m)
    for w in re.findall(r"[a-z]+", low):
        if w in PERSON_NOUNS and not any(w in c.text for c in clues):
            clues.append(Clue(w, "person", w))
            covered.append(w)
            break
    if not any(c.kind == "object" for c in clues):
        for w in re.findall(r"[a-z]+", low):
            if w in OBJECT_NOUNS and not any(w in c.text for c in clues):
                clues.append(Clue(w, "object", w))
                covered.append(w)
                break
    clues = dedupe_clues(clues)
    if len(clues) >= 2:
        return clues
    if clues:
        return [Clue(text)] if len(text) > 1 else clues
    return [Clue(text)] if text else []


def parse_fallback(description: str) -> list[Clue]:
    if ";" in description or "," in description:
        parts = [p.strip().removeprefix("and ").strip()
                 for p in description.replace(";", ",").split(",")]
        parts = [p for p in parts if len(p) > 1]
        if len(parts) >= 2:
            return [_classify_part(p) for p in parts]
        if parts:
            return _parse_sentence(description)
    if not description.strip():
        return []
    return _parse_sentence(description)


def _relaxed_text(clue: Clue) -> str | None:
    if clue.kind == "colour":
        for colour in COLOUR_WORDS:
            out = re.sub(rf"\b{colour}\b", "", clue.text, flags=re.I).strip()
            out = " ".join(out.split())
            if out and out.lower() != clue.text.lower():
                return out
        ref = clue.referent.strip()
        return ref if ref and ref.lower() != clue.text.lower() else None
    if clue.kind == "count":
        out = re.sub(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                     "", clue.text, count=1, flags=re.I).strip()
        out = " ".join(out.split())
        return out if out and out.lower() != clue.text.lower() else None
    return None


def recall_hypotheses(description: str, clues: list[Clue],
                      substitution_colours: tuple[str, ...] | None = None,
                      max_substitutions: int = 2) -> list[Hypothesis]:
    """The user's own sentence first, then the clue branches: the
    sentence is the query plain search would run, so it is never lost."""
    raw = Hypothesis(RAW_LABEL, description, None)
    rest = hypotheses(clues, substitution_colours, max_substitutions)
    return [raw] + [h for h in rest if h.query_text != description]


def hypotheses(clues: list[Clue],
               substitution_colours: tuple[str, ...] | None = None,
               max_substitutions: int = 2) -> list[Hypothesis]:
    full = "; ".join(c.text for c in clues)
    out = [Hypothesis("original", full, None)]
    if len(clues) < 2 and not any(_relaxed_text(c) for c in clues):
        if not substitution_colours:
            return out
    for i in range(len(clues)):
        if len(clues) >= 2:
            rest = [c.text for j, c in enumerate(clues) if j != i]
            out.append(Hypothesis(f"drop:{i}", "; ".join(rest), i))
        relaxed = _relaxed_text(clues[i])
        if relaxed:
            parts = [relaxed if j == i else c.text for j, c in enumerate(clues)]
            if len(clues) >= 2 or relaxed != full:
                out.append(Hypothesis(f"relax:{i}", "; ".join(parts), None))
        if substitution_colours and clues[i].kind == "colour":
            claimed = next((w for w in COLOUR_WORDS if w in clues[i].text.lower()), None)
            subs = [c for c in substitution_colours if c != claimed][:max_substitutions]
            for alt in subs:
                if claimed:
                    new_text = re.sub(rf"\b{claimed}\b", alt, clues[i].text, flags=re.I)
                    parts = [new_text if j == i else c.text for j, c in enumerate(clues)]
                    out.append(Hypothesis(f"sub:{i}:{alt}", "; ".join(parts), None))
    seen: set[str] = set()
    uniq: list[Hypothesis] = []
    for h in out:
        if h.query_text not in seen:
            seen.add(h.query_text)
            uniq.append(h)
    return uniq
