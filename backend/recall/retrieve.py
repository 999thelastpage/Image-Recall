"""Exact cosine retrieval over L2-normalised embeddings (NumPy).
Exact search is preferred until measured scale requires otherwise."""

from __future__ import annotations

import numpy as np


class ExactIndex:
    def __init__(self, image_ids: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(image_ids):
            raise ValueError("embeddings rows must match image_ids")
        self.image_ids = list(image_ids)
        self.embeddings = embeddings.astype(np.float32)

    def __len__(self) -> int:
        return len(self.image_ids)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return [(image_id, cosine)] sorted by score desc, then id
        asc for deterministic tie-breaking."""
        if query.ndim == 1:
            query = query[None, :]
        scores = (self.embeddings @ query[0].astype(np.float32)).ravel()
        order = sorted(
            range(len(scores)), key=lambda i: (-float(scores[i]), self.image_ids[i])
        )
        return [(self.image_ids[i], float(scores[i])) for i in order[:top_k]]

    def rank_of(self, query: np.ndarray, image_id: str) -> int | None:
        """1-based rank of an image under a query (evaluation only;
        never called from the search session path)."""
        if image_id not in self.image_ids:
            return None
        results = self.search(query, len(self.image_ids))
        for rank, (iid, _score) in enumerate(results, start=1):
            if iid == image_id:
                return rank
        return None
