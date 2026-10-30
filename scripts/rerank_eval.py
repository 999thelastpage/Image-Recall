"""Rerank eval: does Muse rescoring at finish improve Hit@5?

Reuses the G2 wrong-clue cases (colour + object). For each case:
session with simulated truthful user -> rank before rerank ->
Muse rerank of the top-10 -> rank after. Reports Hit@5 deltas.

    .venv/bin/python -m scripts.rerank_eval [--n-per-kind 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g2_eval import pick_cases, simulate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=6)
    args = ap.parse_args()

    import numpy as np

    from backend.recall.clues import Clue
    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import read_manifest
    from backend.recall.providers import MuseClient, load_dotenv
    from backend.recall.rerank import MuseReranker
    from backend.recall.retrieve import ExactIndex
    from backend.recall.session import SearchSession

    load_dotenv()
    entries = read_manifest(Path("data/manifests/coco_val2017.jsonl"))
    index = ExactIndex([e.image_id for e in entries],
                       np.load("embeddings/coco_val2017.npy"))
    caps = json.loads(Path("data/coco/captions.json").read_text())
    by_file = {Path(e.path).name: e.image_id for e in entries}
    paths = {e.image_id: e.path for e in entries}
    cases = [c for c in pick_cases(caps, by_file) if c["kind"] != "accurate"]
    # pick_cases spreads 10/10; trim per kind
    trimmed, seen = [], {}
    for c in cases:
        if seen.get(c["kind"], 0) < args.n_per_kind:
            trimmed.append(c)
            seen[c["kind"]] = seen.get(c["kind"], 0) + 1
    cases = trimmed
    print(f"{len(cases)} wrong-clue cases")

    enc = MetaClipEncoder()
    muse = MuseClient()
    reranker = MuseReranker(muse)
    before_hit = after_hit = 0
    for i, case in enumerate(cases, 1):
        target = by_file[case["file"]]
        chips = muse.parse(case["query"])
        clues = [Clue(c["text"], c.get("kind", "other"), c.get("referent", ""))
                 for c in chips if len(c["text"].strip()) > 1]
        s = SearchSession(index, enc, case["query"], clues)
        s.start()
        simulate(s, case)
        s.finish()
        rank_before = next((r for r, c in enumerate(s.candidates, 1)
                            if c.image_id == target), None)
        out = reranker.rerank(s.candidates, paths, case["query"])
        rank_after = next((r for r, c in enumerate(out, 1)
                           if c.image_id == target), None)
        before_hit += rank_before is not None and rank_before <= 5
        after_hit += rank_after is not None and rank_after <= 5
        print(f"[{i:2d}] {case['kind']:6s} {str(rank_before):>5s} -> {str(rank_after):>5s} "
              f"| {case['query'][:55]}")
    n = len(cases)
    print(f"\nHit@5 before rerank {before_hit}/{n} -> after {after_hit}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
