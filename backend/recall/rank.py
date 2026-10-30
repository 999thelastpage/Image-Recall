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


def branch_union(
    index: ExactIndex,
    branch_queries: list[tuple[Hypothesis, np.ndarray]],
    per_branch_k: int = BRANCH_TOP_K,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for hyp, query in branch_queries:
        for image_id, score in index.search(query, per_branch_k):
            cand = merged.get(image_id)
            if cand is None:
                merged[image_id] = Candidate(image_id, score, [hyp.label])
            else:
                cand.score = max(cand.score, score)
                if hyp.label not in cand.branches:
                    cand.branches.append(hyp.label)
    return sorted(merged.values(), key=lambda c: (-c.score, c.image_id))
