"""Gate G1: wrong-clue recovery on 10 hand-made cases.

For 10 COCO images whose caption names a colour, corrupt exactly
that one colour. Baseline = corrupted description searched plainly.
Repair = Muse parses it into clue chips; original + leave-one-out
branches retrieve with reserved quotas; merged union is checked.

Gate (sprint): union contains the target in >= 7/10 cases.
The target id is used ONLY here, for scoring — never in search.

    .venv/bin/python -m scripts.g1_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
           "pink", "brown", "black", "white", "gray", "gold"]
WRONG = {"red": "blue", "orange": "purple", "yellow": "green",
         "green": "red", "blue": "orange", "purple": "yellow",
         "pink": "brown", "brown": "black", "black": "white",
         "white": "black", "gray": "pink", "gold": "gray"}


def pick_cases(captions: dict[str, str], n: int = 10) -> list[dict]:
    cases = []
    for fname in sorted(captions):
        cap = captions[fname]
        low = f" {cap.lower()} "
        hit = next((c for c in COLOURS if f" {c} " in low), None)
        if hit:
            import re

            wrong = WRONG[hit]
            corrupted = re.sub(
                rf"\b{hit}\b",
                lambda m: wrong.capitalize() if m.group(0)[0].isupper() else wrong,
                cap, count=1, flags=re.IGNORECASE,
            )
            cases.append({
                "file": fname, "caption": cap, "colour": hit,
                "corrupted": corrupted,
            })
        if len(cases) >= n * 3:  # keep a pool, take an even spread
            break
    step = max(1, len(cases) // n)
    return cases[::step][:n]


def main() -> int:
    import numpy as np

    from backend.recall.clues import Clue, hypotheses
    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import read_manifest
    from backend.recall.providers import MuseClient, load_dotenv
    from backend.recall.rank import branch_union
    from backend.recall.retrieve import ExactIndex

    load_dotenv()
    entries = read_manifest(Path("data/manifests/coco_val2017.jsonl"))
    mat = np.load("embeddings/coco_val2017.npy")
    captions = json.loads(Path("data/coco/captions.json").read_text())
    by_file = {Path(e.path).name: e.image_id for e in entries}
    index = ExactIndex([e.image_id for e in entries], mat)

    cases = pick_cases(captions)
    print(f"{len(cases)} wrong-colour cases (one clue corrupted each)")
    enc = MetaClipEncoder()
    muse = MuseClient()

    contained = baseline_top5 = 0
    for i, case in enumerate(cases, 1):
        target = by_file[case["file"]]
        # Baseline: plain search with the corrupted description.
        q_base = enc.encode_texts([case["corrupted"]])
        base_rank = index.rank_of(q_base, target)
        # Repair: parse -> hypotheses -> branch union.
        chips = muse.parse(case["corrupted"])
        clues = [Clue(c["text"]) for c in chips]
        hyps = hypotheses(clues)
        queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
        union = branch_union(index, queries, per_branch_k=50)
        union_rank = next(
            (r for r, c in enumerate(union, 1) if c.image_id == target), None
        )
        via = next((c.branches for c in union if c.image_id == target), [])
        contained += union_rank is not None
        baseline_top5 += base_rank is not None and base_rank <= 5
        print(f"[{i}] {case['colour']}->{WRONG[case['colour']]} "
              f"baseline_rank={base_rank} union_rank={union_rank} via={via} "
              f"| {case['corrupted'][:64]}")

    n = len(cases)
    print(f"\nGATE G1: union contains target {contained}/{n} "
          f"(need >=7) | baseline Hit@5 {baseline_top5}/{n}")
    print("G1 PASS" if contained >= 7 else "G1 FAIL")
    return 0 if contained >= 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
