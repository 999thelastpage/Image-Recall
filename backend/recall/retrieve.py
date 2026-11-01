"""Exact cosine retrieval over L2-normalised embeddings (NumPy).

Exact search is preferred until measured scale requires otherwise.
Top-k selection uses argpartition plus a tie-complete boundary, so
an 83k-row scan never pays for a full sort of all 83k scores, and
batched queries share one matrix multiply (one read of the matrix).
"""

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

    def _rank(self, scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Top-k of one score row: score desc, then image_id asc.

        Deterministic even at the cut: every row whose score ties the
        k-th score is considered, so no id-ascending tie is lost to
        argpartition's arbitrary tie subset."""
        n = len(scores)
        k = min(max(top_k, 0), n)
        if k == 0:
            return []
        if k < n:
            part = np.argpartition(scores, n - k)[n - k:]
            threshold = scores[part].min()
            above = np.flatnonzero(scores > threshold).tolist()
            tied = sorted(
                (int(i) for i in np.flatnonzero(scores == threshold)),
                key=lambda i: self.image_ids[i],
            )
            cand = above + tied[: k - len(above)]
        else:
            cand = list(range(n))
        order = sorted(
            cand, key=lambda i: (-float(scores[i]), self.image_ids[i])
        )
        return [(self.image_ids[i], float(scores[i])) for i in order]

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return [(image_id, cosine)] sorted by score desc, then id
        asc for deterministic tie-breaking."""
        if query.ndim == 1:
            query = query[None, :]
        scores = (self.embeddings @ query[0].astype(np.float32)).ravel()
        return self._rank(scores, top_k)

    def search_many(
        self, queries: np.ndarray, top_k: int
    ) -> list[list[tuple[str, float]]]:
        """One top-k list per query row, from a single matmul — the
        embedding matrix is read once for all branches, not per branch."""
        if queries.ndim == 1:
            queries = queries[None, :]
        all_scores = self.embeddings @ queries.astype(np.float32).T
        return [
            self._rank(all_scores[:, j], top_k) for j in range(all_scores.shape[1])
        ]

    def score_rows(self, query: np.ndarray, rows: list[int]) -> np.ndarray:
        """Cosine of one query against chosen embedding rows — used to
        rescore every union candidate under a revised query, so no
        candidate is invisible just because a top-k cut excluded it."""
        q = np.asarray(query, dtype=np.float32).ravel()
        return self.embeddings[rows] @ q

    def score_stats(self, query: np.ndarray) -> tuple[float, float]:
        """Mean and std of a query's cosines over a sample of the
        corpus, so scores from different branches can be compared."""
        q = np.asarray(query, dtype=np.float32).ravel()
        scores = self.embeddings[:: max(1, len(self) // 5000)] @ q
        return float(scores.mean()), float(scores.std()) or 1.0

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
