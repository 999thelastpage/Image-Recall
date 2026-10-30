"""Union-recall probe: at a given per-branch quota, how often does
the leave-one-out union contain the target at all? This is the
coverage ceiling — ranking/questions can never beat it.

Compares quotas on wrong-colour cases from train2014 (83k).

    .venv/bin/python -m scripts.union_recall [--n 10] [--quotas 50,150]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g1_eval import WRONG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--quotas", default="50,150")
    ap.add_argument("--corpus", default="train2014")
    ap.add_argument("--captions", default="data/coco/captions_train2014.json")
    args = ap.parse_args()
    quotas = [int(q) for q in args.quotas.split(",")]

    import re

    import numpy as np

    from backend.recall.clues import hypotheses, parse_fallback
    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import read_manifest
    from backend.recall.rank import branch_union
    from backend.recall.retrieve import ExactIndex

    entries = read_manifest(Path(f"data/manifests/{args.corpus}.jsonl"))
    index = ExactIndex([e.image_id for e in entries],
                       np.load(f"embeddings/{args.corpus}.npy"))
    caps = json.loads(Path(args.captions).read_text())
    by_file = {Path(e.path).name: e.image_id for e in entries}

    colours = list(WRONG)
    cases = []
    for fname in sorted(caps):
        cap = caps[fname]
        hit = next((c for c in colours if re.search(rf"\b{c}\b", cap, re.I)), None)
        if hit and fname in by_file:
            corrupted = re.sub(rf"\b{hit}\b", WRONG[hit], cap, count=1, flags=re.I)
            cases.append((fname, corrupted))
        if len(cases) >= args.n * 3:
            break
    step = max(1, len(cases) // args.n)
    cases = cases[::step][: args.n]
    print(f"{len(cases)} wrong-colour cases on {args.corpus} ({len(index)} photos)")

    from backend.recall.providers import MuseClient, load_dotenv

    load_dotenv()
    enc = MetaClipEncoder()
    muse = MuseClient()
    parsed = []
    for fname, query in cases:
        chips = muse.parse(query)
        from backend.recall.clues import Clue

        clues = [Clue(c["text"], c.get("kind", "other"), c.get("referent", ""))
                 for c in chips if len(c["text"].strip()) > 1] or parse_fallback(query)
        parsed.append((by_file[fname], hypotheses(clues)))

    for quota in quotas:
        contained = 0
        for target, hyps in parsed:
            queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
            union = branch_union(index, queries, per_branch_k=quota)
            contained += any(c.image_id == target for c in union)
        print(f"quota {quota:4d}: union contains target {contained}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
