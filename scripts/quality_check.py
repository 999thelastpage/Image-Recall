"""Corpus quality check: for N sampled images, ask for the photo
in its own caption's words and see where it lands. This is the
honest ceiling test of the current (full-resolution) approach —
harder as the corpus grows, because distractors multiply.

    .venv/bin/python -m scripts.quality_check --corpus train2014 --n 100

Reports Recall@1/5/10/50, median rank, mean query latency, and the
5 worst misses with captions so a human can judge failure layers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="train2014")
    ap.add_argument("--captions", default="data/coco/captions_train2014.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=None,
                    help="HF model id; embeddings cached per model under embeddings/")
    args = ap.parse_args()

    import numpy as np

    from backend.recall.config import ENCODER_MODEL
    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import read_manifest
    from backend.recall.retrieve import ExactIndex

    entries = read_manifest(Path(f"data/manifests/{args.corpus}.jsonl"))
    model = args.model or ENCODER_MODEL
    slug = model.replace("/", "_")
    cache = Path(f"embeddings/{args.corpus}__{slug}.npy")
    default_cache = Path(f"embeddings/{args.corpus}.npy")
    enc = MetaClipEncoder(model)
    if cache.exists():
        mat = np.load(cache)
    elif args.model is None and default_cache.exists():
        mat = np.load(default_cache)
        cache = default_cache
    else:
        import time as _time

        # Chunked + resumable: a killed run keeps finished chunks.
        chunk_dir = Path(f"embeddings/{args.corpus}__{slug}.chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk = 2000
        paths = [e.path for e in entries]
        t0 = _time.time()
        done = 0
        for i in range(0, len(paths), chunk):
            part = chunk_dir / f"{i:07d}.npy"
            if not part.exists():
                np.save(part, enc.encode_images(paths[i : i + chunk]))
            done = min(i + chunk, len(paths))
            print(f"  embedded {done}/{len(paths)}", flush=True)
        mat = np.concatenate(
            [np.load(chunk_dir / f"{i:07d}.npy") for i in range(0, len(paths), chunk)]
        )
        np.save(cache, mat)
        dt = _time.time() - t0
        print(f"embedded {len(entries)} with {model} in {dt:.0f}s "
              f"({len(entries) / max(dt, 1):.1f} img/s) -> {cache}")
    print(f"model={model} dim={mat.shape[1]}")
    caps = json.loads(Path(args.captions).read_text())
    by_file = {Path(e.path).name: e.image_id for e in entries}
    usable = [(f, by_file[f]) for f in sorted(caps) if f in by_file]
    step = max(1, len(usable) // args.n)
    sample = usable[::step][: args.n]
    print(f"corpus={args.corpus} size={len(entries)} sampled={len(sample)}")

    index = ExactIndex([e.image_id for e in entries], mat)
    hits = {1: 0, 5: 0, 10: 0, 50: 0}
    ranks: list[int] = []
    lat: list[float] = []
    misses: list[tuple[int, str, str]] = []
    for fname, target in sample:
        t0 = time.time()
        q = enc.encode_texts([caps[fname]])
        rank = index.rank_of(q, target)
        lat.append((time.time() - t0) * 1000)
        assert rank is not None
        ranks.append(rank)
        for k in hits:
            hits[k] += rank <= k
        misses.append((rank, fname, caps[fname]))
    n = len(sample)
    ranks.sort()
    print(json.dumps({
        "recall@1": round(hits[1] / n, 3), "recall@5": round(hits[5] / n, 3),
        "recall@10": round(hits[10] / n, 3), "recall@50": round(hits[50] / n, 3),
        "median_rank": ranks[n // 2], "mean_query_ms": round(sum(lat) / n, 1),
    }, indent=1))
    print("\nworst 5 misses:")
    for rank, fname, cap in sorted(misses, reverse=True)[:5]:
        print(f"  rank={rank} {fname} | {cap[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
