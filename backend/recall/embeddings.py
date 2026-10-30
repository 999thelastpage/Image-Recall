"""Image/text embedding encoders.

Dinov2Encoder is the real path (Meta open weights). StubEncoder is a
deterministic stand-in for tests and plumbing demos ONLY — its
scores say nothing about retrieval quality and output is labelled
as such wherever it is used.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .config import EMBEDDING_DIM, ENCODER_MODEL


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class StubEncoder:
    """Deterministic hash-seeded vectors. Tests/plumbing only."""

    name = "stub"
    is_stub = True

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _vec(self, key: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def encode_images(self, paths: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(p) for p in paths]))


class Dinov2Encoder:
    """facebook/dinov2-base via transformers. Lazy heavy imports so
    the rest of the package runs without torch installed."""

    name = ENCODER_MODEL
    is_stub = False

    def __init__(self, model_name: str = ENCODER_MODEL) -> None:
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        torch = self._torch
        vecs = []
        with torch.no_grad():
            for path in paths:
                img = Image.open(path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt")
                out = self.model(**inputs)
                vecs.append(out.last_hidden_state[:, 0].cpu().numpy())
        return _normalize(np.concatenate(vecs))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "DINOv2 has no text tower; text queries are embedded via "
            "Muse Spark reranking over the image index (see providers)."
        )
