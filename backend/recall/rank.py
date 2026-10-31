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
from .config import BRANCH_TOP_K
from .retrieve import ExactIndex


@dataclass
class Candidate:
    image_id: str
    score: float
    branches: list[str] = field(default_factory=list)
    branch_scores: dict[str, float] = field(default_factory=dict)


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
        for image_id, score in hits:
            cand = merged.get(image_id)
            if cand is None:
                merged[image_id] = Candidate(
                    image_id, score, [hyp.label], {hyp.label: score}
                )
            else:
                cand.score = max(cand.score, score)
                cand.branch_scores[hyp.label] = max(
                    score, cand.branch_scores.get(hyp.label, score)
                )
                if hyp.label not in cand.branches:
                    cand.branches.append(hyp.label)
    return sorted(merged.values(), key=lambda c: (-c.score, c.image_id))


def weighted_rerank(
    candidates: list[Candidate], weights: dict[str, float]
) -> list[Candidate]:
    """Soft re-ranking: final = max over branches of weight*score.
    No candidate is ever removed (no hard filters); a branch the
    user doubted simply speaks louder. Deterministic."""
    out = []
    for c in candidates:
        best = max(
            (weights.get(b, 1.0) * s for b, s in c.branch_scores.items()),
            default=c.score,
        )
        # Floor: an answer may reorder, but it must never lower a
        # candidate's own score — demotion by omission was measured
        # pushing targets from #2 to #52.
        best = max(best, c.score)
        out.append(Candidate(c.image_id, best, list(c.branches), dict(c.branch_scores)))
    return sorted(out, key=lambda c: (-c.score, c.image_id))
