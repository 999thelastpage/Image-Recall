"""Clue parsing (fallback) and leave-one-clue-out hypotheses.

The primary parser is Muse Spark (see providers). This structured
fallback always works offline: clues separated by ';'. It never
guesses at grammar it cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clue:
    text: str
    kind: str = "other"  # colour|object|count|setting|activity|person|other
    referent: str = ""


@dataclass(frozen=True)
class Hypothesis:
    label: str  # "original" or "drop:<index>"
    query_text: str
    dropped_index: int | None  # None for the original


def dedupe_clues(clues: list[Clue]) -> list[Clue]:
    """Drop exact duplicates and any clue whose text is contained
    in a longer clue ('mini orange train' swallows 'train').
    Order preserved; longest phrasing wins."""
    seen: set[str] = set()
    uniq: list[Clue] = []
    for c in clues:
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


def parse_fallback(description: str) -> list[Clue]:
    parts = [p.strip() for p in description.split(";")]
    clues = [Clue(p) for p in parts if p]
    if not clues and description.strip():
        clues = [Clue(description.strip())]
    return clues


def hypotheses(clues: list[Clue]) -> list[Hypothesis]:
    """Original plus one variant per dropped clue. With a single
    clue there is nothing to drop, so only the original is returned."""
    full = "; ".join(c.text for c in clues)
    out = [Hypothesis("original", full, None)]
    if len(clues) < 2:
        return out
    for i in range(len(clues)):
        rest = [c.text for j, c in enumerate(clues) if j != i]
        out.append(Hypothesis(f"drop:{i}", "; ".join(rest), i))
    return out
