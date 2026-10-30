"""G0 encoder smoke: embed N COCO images on GPU, query with their
own captions, report Recall@5/@10/@50 + throughput.

Gate (handoff): encoder usable. A model that cannot put an image
in the top-50 for its own caption is not a retrieval encoder.

Usage (from repo root, in a shell with GPU access):
    .venv/bin/python -m scripts.g0_encoder --n 500
Requires data/coco/val2017/*.jpg and data/coco/captions.json
({file_name: caption}) — see scripts/fetch_coco.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_CANDIDATES = [
    "facebook/metaclip-b32-fullcc2.5b",
    "facebook/metaclip-h14-fullcc2.5b",
]


def _feats(out):
    """transformers 5.x returns a pooled-output object; older returns a tensor."""
    import numpy as np

    tensor = getattr(out, "pooler_output", out)
    return tensor.cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    img_dir = Path("data/coco/val2017")
    caps = json.loads(Path("data/coco/captions.json").read_text())
    files = sorted(p for p in img_dir.glob("*.jpg") if p.name in caps)[: args.n]
    if len(files) < 50:
        print(f"only {len(files)} captioned images found — run scripts/fetch_coco.sh")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = [args.model] if args.model else MODEL_CANDIDATES
    model = processor = used = None
    for cand in candidates:
        try:
            processor = AutoProcessor.from_pretrained(cand)
            model = CLIPModel.from_pretrained(cand).to(device).eval()
            used = cand
            break
        except Exception as e:  # noqa: BLE001
            print(f"skip {cand}: {type(e).__name__} {str(e)[:120]}")
    if model is None:
        return 1
    print(f"model={used} device={device} images={len(files)}")

    def embed_images(paths: list[Path]) -> np.ndarray:
        vecs = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(paths), args.batch):
                imgs = [Image.open(p).convert("RGB") for p in paths[i : i + args.batch]]
                inp = processor(images=imgs, return_tensors="pt").to(device)
                vecs.append(_feats(model.get_image_features(**inp)))
        dt = time.time() - t0
        mat = np.concatenate(vecs).astype(np.float32)
        mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
        print(f"images: {len(paths)} in {dt:.1f}s = {len(paths)/dt:.1f} img/s")
        return mat

    def embed_texts(texts: list[str]) -> np.ndarray:
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch):
                inp = processor(text=texts[i : i + args.batch], return_tensors="pt",
                                padding=True, truncation=True).to(device)
                vecs.append(_feats(model.get_text_features(**inp)))
        mat = np.concatenate(vecs).astype(np.float32)
        mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
        return mat

    from backend.recall.retrieve import ExactIndex

    ids = [f.stem for f in files]
    index = ExactIndex(ids, embed_images(files))
    queries = embed_texts([caps[f.name] for f in files])
    hits = {5: 0, 10: 0, 50: 0}
    for qi, q in enumerate(queries):
        rank = index.rank_of(q, ids[qi])
        for k in hits:
            hits[k] += rank is not None and rank <= k
    n = len(files)
    print(json.dumps({"model": used, "device": device, "n": n,
                      **{f"recall@{k}": round(hits[k] / n, 4) for k in hits}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
