"""Branch union + base scoring.

Each hypothesis branch retrieves with a reserved quota so one
branch cannot crowd out the others; the merged union keeps the best
branch score per image plus provenance (which branches surfaced it).
No hard filters: a low branch score never removes a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clues import Hypothesis
from .config import BRANCH_TOP_K, RRF_K
from .retrieve import ExactIndex


@dataclass
class Candidate:
    image_id: str
    score: float
    branches: list[str] = field(default_factory=list)
    branch_scores: dict[str, float] = field(default_factory=dict)
    fused: float = 0.0  # reciprocal-rank fusion score (scale-free)


def branch_union(
    index: ExactIndex,
    branch_queries: list[tuple[Hypothesis, np.ndarray]],
    per_branch_k: int = BRANCH_TOP_K,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    # Batched: one matmul for every branch (matrix read once).
    if hasattr(index, "search_many") and branch_queries:
        vecs = np.stack(
            [np.asarray(q, dtype=np.float32).ravel() for _h, q in branch_queries]
        )
        per_branch = index.search_many(vecs, per_branch_k)
    else:
        per_branch = [index.search(q, per_branch_k) for _h, q in branch_queries]
    for (hyp, _query), hits in zip(branch_queries, per_branch, strict=True):
        for rank_pos, (image_id, score) in enumerate(hits, start=1):
            rrf = 1.0 / (RRF_K + rank_pos)
            cand = merged.get(image_id)
            if cand is None:
                merged[image_id] = Candidate(
                    image_id, score, [hyp.label], {hyp.label: score}, fused=rrf
                )
            else:
                cand.score = max(cand.score, score)
                cand.fused += rrf
                cand.branch_scores[hyp.label] = max(
                    score, cand.branch_scores.get(hyp.label, score)
                )
                if hyp.label not in cand.branches:
                    cand.branches.append(hyp.label)
    # Calibrated fusion: raw cosine dominates, reciprocal-rank
    # support across branches breaks close calls. Pure RRF would
    # let correlated relax branches outvote one exact branch.
    return sorted(merged.values(),
                  key=lambda c: (-(c.score + 0.15 * c.fused), -c.score, c.image_id))


def weighted_rerank(
    candidates: list[Candidate], weights: dict[str, float]
) -> list[Candidate]:
    """Soft re-ranking: final = max over branches of weight*score.
    No candidate is ever removed and no score ever falls: demotion
    is relative (doubted branches speak louder), because absolute
    demotion by omission was measured pushing targets #2 -> #52."""
    out = []
    for c in candidates:
        best = max(
            (weights.get(b, 1.0) * s for b, s in c.branch_scores.items()),
            default=c.score,
        )
        best = max(best, c.score)
        out.append(Candidate(c.image_id, best, list(c.branches),
                             dict(c.branch_scores), fused=c.fused))
    return sorted(out, key=lambda c: (-c.score, c.image_id))
