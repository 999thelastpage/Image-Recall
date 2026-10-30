"""Terminal demo: description -> clue hypotheses -> branch union.

Real mode (default when the ingested index exists): MetaCLIP
embeddings over the 5k COCO index, Muse Spark clue parsing when
a key is configured, structured fallback otherwise.
Stub mode (--stub): deterministic fixture, labelled as plumbing.

Usage:
    python -m scripts.demo_search "a red jacket on a bicycle at the beach"
    python -m scripts.demo_search --stub "red jacket; beach; daylight"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.recall.clues import Clue, hypotheses, parse_fallback  # noqa: E402
from backend.recall.rank import branch_union  # noqa: E402
from backend.recall.retrieve import ExactIndex  # noqa: E402


def _stub(description: str) -> int:
    from backend.recall.embeddings import StubEncoder

    fixture = Path("data/fixture_captions.json")
    if not fixture.exists():
        raise SystemExit("fixture missing: run `python -m scripts.make_fixture` first")
    items = json.loads(fixture.read_text(encoding="utf-8"))
    enc = StubEncoder()
    index = ExactIndex(
        [it["image_id"] for it in items],
        enc.encode_texts([it["caption"] for it in items]),
    )
    clues = parse_fallback(description)
    hyps = hypotheses(clues)
    print("[STUB ENCODER — plumbing demo, not retrieval quality]")
    print(f"description: {description!r} -> {len(clues)} clue(s), {len(hyps)} branch(es)")
    queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
    for rank, cand in enumerate(branch_union(index, queries, per_branch_k=50)[:5], 1):
        print(f"  #{rank} {cand.image_id} score={cand.score:.4f} via {cand.branches}")
    return 0


def _real(description: str) -> int:
    import numpy as np

    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import read_manifest
    from backend.recall.providers import MuseClient, ProviderDisabled, load_dotenv

    load_dotenv()
    entries = read_manifest(Path("data/manifests/coco_val2017.jsonl"))
    index = ExactIndex(
        [e.image_id for e in entries], np.load("embeddings/coco_val2017.npy")
    )
    enc = MetaClipEncoder()
    try:
        chips = MuseClient().parse(description)
        clues = [Clue(c["text"]) for c in chips]
        source = "Muse Spark parse"
    except (ProviderDisabled, Exception) as e:  # fall back, never block
        clues = parse_fallback(description)
        source = f"fallback parse ({type(e).__name__})"
    hyps = hypotheses(clues)
    print(f"[REAL index: {len(index)} images | {source}]")
    print(f"description: {description!r} -> clues {[c.text for c in clues]}")
    base = index.search(enc.encode_texts([description]), 5)
    print("baseline (plain search):")
    for rank, (iid, score) in enumerate(base, 1):
        print(f"  #{rank} {iid} {score:.4f}")
    queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
    print("branch union (original + leave-one-clue-out):")
    for rank, cand in enumerate(branch_union(index, queries, per_branch_k=50)[:5], 1):
        print(f"  #{rank} {cand.image_id} {cand.score:.4f} via {cand.branches}")
    return 0


def main(argv: list[str]) -> int:
    stub = "--stub" in argv
    argv = [a for a in argv if a != "--stub"]
    if not argv:
        print(__doc__)
        return 2
    if stub or not Path("embeddings/coco_val2017.npy").exists():
        return _stub(argv[0])
    return _real(argv[0])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
