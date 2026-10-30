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

RERANK_WEIGHT = 0.45
RERANK_TOP_N = 10

_PROMPT = (
    "A user is looking for a photo they described as: \"{desc}\"\n"
    "Score how well this image matches that description, 0 (no "
    "match) to 10 (exact photo). Judge objects, colours, counts, "
    "setting and relations. Return ONLY JSON: {{\"score\": <0-10>}}"
)


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


class MuseReranker:
    def __init__(self, muse_client, max_workers: int = 4) -> None:
        self.client = muse_client
        self.max_workers = max_workers

    def _score(self, args) -> float | None:
        path, description = args
        try:
            txt = self.client.chat_with_image(
                path, _PROMPT.format(desc=description)
            )
            data = json.loads(txt[txt.index("{") : txt.rindex("}") + 1])
            return max(0.0, min(10.0, float(data["score"]))) / 10.0
        except Exception:  # noqa: BLE001 — a failed score is neutral
            return None

    def rerank(
        self,
        candidates: list[Candidate],
        image_paths: dict[str, str],
        description: str,
        top_n: int = RERANK_TOP_N,
        weight: float = RERANK_WEIGHT,
    ) -> list[RerankedCandidate]:
        head, tail = candidates[:top_n], candidates[top_n:]
        enc_norm = _minmax([c.score for c in head])
        jobs = [(image_paths[c.image_id], description) for c in head]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            muse_scores = list(ex.map(self._score, jobs))
        out: list[RerankedCandidate] = []
        for cand, enc_n, ms in zip(head, enc_norm, muse_scores, strict=True):
            muse_part = ms if ms is not None else enc_n  # neutral fallback
            final = (1 - weight) * enc_n + weight * muse_part
            out.append(RerankedCandidate(cand.image_id, final, cand.score, ms,
                                         list(cand.branches)))
        out.sort(key=lambda r: (-r.final, r.image_id))
        out.extend(
            RerankedCandidate(c.image_id, -1.0, c.score, None, list(c.branches))
            for c in tail
        )
        return out
