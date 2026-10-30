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


def _feats(out) -> np.ndarray:
    tensor = getattr(out, "pooler_output", out)
    return tensor.cpu().numpy().astype(np.float32)


class MetaClipEncoder:
    """MetaCLIP (Meta open weights, CLIP-style joint text+image
    space). Lazy heavy imports so the package runs without torch."""

    name = ENCODER_MODEL
    is_stub = False

    def __init__(self, model_name: str = ENCODER_MODEL, device: str | None = None) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor, CLIPModel  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def encode_images(self, paths: list[str], batch: int = 64) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        torch = self._torch
        vecs = []
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
                inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
                vecs.append(_feats(self.model.get_image_features(**inp)))
        return _normalize(np.concatenate(vecs))

    def encode_texts(self, texts: list[str], batch: int = 64) -> np.ndarray:
        torch = self._torch
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                inp = self.processor(
                    text=texts[i : i + batch], return_tensors="pt",
                    padding=True, truncation=True,
                ).to(self.device)
                vecs.append(_feats(self.model.get_text_features(**inp)))
        return _normalize(np.concatenate(vecs))
