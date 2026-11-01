"""Muse Spark reranking of the top candidates.

The encoder is fast and broad; Muse is slow and precise. At
finish time, Muse scores the top-N candidates against the
original description (it handles relations and attributes the
encoder mangles), and we blend: final = (1-w)*encoder_norm +
w*muse. Candidates outside the top-N keep their order after the
reranked block. Deterministic given the same Muse scores.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .rank import Candidate

RERANK_WEIGHT = 0.65
RERANK_TOP_N = 40
GUARD_TOP = 5    # union candidates protected from deep demotion
GUARD_FLOOR = 10  # they cannot fall below this rank on a weak Muse score
GUARD_MIN_MUSE = 0.1  # at or below this, Muse may demote freely

_PROMPT = (
    "A user is looking for a photo they described as: \"{desc}\"\n"
    "Score how well this image matches that description, 0 (no "
    "match) to 10 (exact photo). Judge objects, colours, counts, "
    "setting and relations. Return ONLY JSON: {{\"score\": <0-10>}}"
)


def doubt_aware_description(
    description: str,
    dropped: list[str],
    corrections: list[tuple[str, str]],
    notes: list[str] | None = None,
) -> str:
    """The description Muse scores against must reflect what the
    user doubted and corrected: scoring against the original wrong
    description penalises the true photo for the user's own error."""
    parts = [description]
    if dropped:
        parts.append(
            "Note: the user may be wrong about: " + ", ".join(dropped)
            + ". Do not penalise a photo for differing on those details."
        )
    for old, new in corrections:
        parts.append(f"The user corrected '{old}' to '{new}'.")
    if notes:
        parts.extend(notes)
    return " ".join(parts)


@dataclass
class RerankedCandidate:
    image_id: str
    final: float
    encoder: float
    muse: float | None
    branches: list[str]


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _znorm(values: list[float]) -> list[float]:
    """Per-query z-normalised scores mapped to [0, 1].

    Min-max lets one outlier compress every other candidate;
    z-norm with clipping keeps the blend calibrated when the
    encoder scores are tightly clustered."""
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    if std < 1e-9:
        return [0.5] * len(values)
    return [min(1.0, max(0.0, ((v - mean) / std + 2.0) / 4.0)) for v in values]


class ScoreTable:
    """Scores made in advance, in the form MuseReranker reads, so a backup
    model uses the same blend and demotion guard."""

    def __init__(self, model: str, logits: dict[str, float]) -> None:
        self.model = model
        paths = list(logits)
        self.scores = dict(zip(paths, _znorm([logits[p] for p in paths]), strict=True))

    def chat_with_image(self, path: str, prompt: str, max_tokens: int = 800) -> str:
        return json.dumps({"score": self.scores[path] * 10})


class MuseReranker:
    def __init__(self, muse_client, max_workers: int = 4,
                 cache: dict | None = None) -> None:
        self.client = muse_client
        self.max_workers = max_workers
        self.cache = cache  # shared scores: a repeat check costs nothing

    def _score(self, args) -> float | None:
        path, description = args
        key = (getattr(self.client, "model", ""), path, description)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        try:
            txt = self.client.chat_with_image(
                path, _PROMPT.format(desc=description)
            )
            data = json.loads(txt[txt.index("{") : txt.rindex("}") + 1])
            score = max(0.0, min(10.0, float(data["score"]))) / 10.0
            if self.cache is not None:
                self.cache[key] = score
            return score
        except Exception:  # noqa: BLE001: a failed score is neutral
            return None

    def rerank(
        self,
        candidates: list[Candidate],
        image_paths: dict[str, str],
        description: str,
        top_n: int = RERANK_TOP_N,
        weight: float = RERANK_WEIGHT,
        colour_matches: dict[str, bool | None] | None = None,
    ) -> list[RerankedCandidate]:
        head, tail = candidates[:top_n], candidates[top_n:]
        enc_norm = _znorm([c.score for c in head])
        jobs = [(image_paths[c.image_id], description) for c in head]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            muse_scores = list(ex.map(self._score, jobs))
        out: list[RerankedCandidate] = []
        for cand, enc_n, ms in zip(head, enc_norm, muse_scores, strict=True):
            muse_part = ms if ms is not None else enc_n  # neutral fallback
            final = (1 - weight) * enc_n + weight * muse_part
            if colour_matches is not None:
                verdict = colour_matches.get(cand.image_id)
                if verdict is True:
                    final += 0.05
                elif verdict is False:
                    final -= 0.05  # soft only; None abstains
            out.append(RerankedCandidate(cand.image_id, final, cand.score, ms,
                                         list(cand.branches)))
        out.sort(key=lambda r: (-r.final, r.image_id))
        # Demotion guard: a union top-5 candidate with a non-trivial
        # Muse score cannot be buried below rank 10 by the blend.
        # Measured failure: union 2 -> final 167 on a middling score.
        orig_pos = {c.image_id: i for i, c in enumerate(head)}
        for r in list(out):
            if orig_pos.get(r.image_id, 99) < GUARD_TOP \
                    and r.muse is not None and r.muse > GUARD_MIN_MUSE:
                pos = out.index(r)
                if pos >= GUARD_FLOOR:
                    out.pop(pos)
                    out.insert(GUARD_FLOOR - 1, r)
        out.extend(
            RerankedCandidate(c.image_id, -1.0, c.score, None, list(c.branches))
            for c in tail
        )
        return out
