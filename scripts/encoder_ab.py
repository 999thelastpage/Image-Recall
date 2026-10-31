"""Encoder A/B gate: ceiling first, wrong-clue repair second.

For each candidate encoder, embed the corpus once (cached per
model), then report true-caption Recall@5/50 (the ceiling) and
wrong-clue union coverage separately. Decide on ceiling lift,
not clean R@1 alone. Stub mode proves the plumbing offline.

    .venv/bin/python -m scripts.encoder_ab --stub
    .venv/bin/python -m scripts.encoder_ab --models a,b --n 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarise(ranks: list[int | None], ks=(5, 50)) -> dict:
    n = len(ranks) or 1
    out = {}
    for k in ks:
        out[f"recall@{k}"] = round(
            sum(1 for r in ranks if r is not None and r <= k) / n, 3)
    finite = sorted(r for r in ranks if r is not None)
    out["median_rank"] = finite[len(finite) // 2] if finite else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated HF ids; default = candidates")
    ap.add_argument("--corpus", default="coco_val2017")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--stub", action="store_true",
                    help="offline plumbing check with StubEncoder")
    args = ap.parse_args()

    import json

    import numpy as np

    from backend.recall.clues import hypotheses, parse_fallback
    from backend.recall.embeddings import ENCODER_CANDIDATES, StubEncoder, build_encoder
    from backend.recall.manifest import read_manifest
    from backend.recall.rank import branch_union
    from backend.recall.retrieve import ExactIndex
    from scripts.g1_eval import WRONG, pick_cases

    entries = read_manifest(Path(f"data/manifests/{args.corpus}.jsonl"))
    caps = json.loads(Path("data/coco/captions.json").read_text())
    by_file = {Path(e.path).name: e.image_id for e in entries}
    cases = pick_cases(caps, n=10)
    models = [StubEncoder()] if args.stub else [
        build_encoder(m) for m in (
            args.models.split(",") if args.models else ENCODER_CANDIDATES[:2])
    ]
    # True-caption sample for the ceiling.
    usable = [(f, by_file[f]) for f in sorted(caps) if f in by_file]
    step = max(1, len(usable) // args.n)
    sample = usable[::step][:args.n]
    for enc in models:
        name = getattr(enc, "name", "stub")
        if getattr(enc, "is_stub", False):
            # Stub photo vectors are caption vectors: plumbing only.
            mat = enc.encode_texts([caps[f] for f, _ in sample])
            ids = [t for _, t in sample]
            index = ExactIndex(ids, mat)
            ceiling = [index.rank_of(enc.encode_texts([caps[f]]), t)
                       for f, t in sample]
        else:
            slug = name.replace("/", "_")
            cache = Path(f"embeddings/{args.corpus}__{slug}.npy")
            mat = np.load(cache) if cache.exists() else None
            if mat is None:
                print(f"{name}: no cached embeddings at {cache}; "
                      "run quality_check --model first")
                continue
            index = ExactIndex([e.image_id for e in entries], mat)
            ceiling = [index.rank_of(enc.encode_texts([caps[f]]), t)
                       for f, t in sample]
        print(name, "ceiling", summarise(ceiling))
        union_ranks: list[int | None] = []
        for case in cases:
            target = by_file[case["file"]]
            clues = parse_fallback(case["corrupted"])
            hyps = hypotheses(clues)
            queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
            union = branch_union(index, queries, per_branch_k=50)
            union_ranks.append(next(
                (r for r, c in enumerate(union, 1) if c.image_id == target), None))
        print(name, "wrong-colour union", summarise(union_ranks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
