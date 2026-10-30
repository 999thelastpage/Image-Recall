"""Terminal vertical slice: description → clue hypotheses →
branch union → ranked results.

Runs on the deterministic fixture with StubEncoder when no real
index exists. Output is labelled STUB so nobody mistakes plumbing
for retrieval quality.

Usage:
    python -m scripts.demo_search "red jacket; beach; daylight"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.recall.clues import hypotheses, parse_fallback  # noqa: E402
from backend.recall.embeddings import StubEncoder  # noqa: E402
from backend.recall.rank import branch_union  # noqa: E402
from backend.recall.retrieve import ExactIndex  # noqa: E402


def build_fixture_index() -> tuple[ExactIndex, StubEncoder]:
    fixture = Path("data/fixture_captions.json")
    if not fixture.exists():
        raise SystemExit("fixture missing: run `python -m scripts.make_fixture` first")
    items = json.loads(fixture.read_text(encoding="utf-8"))
    enc = StubEncoder()
    index = ExactIndex(
        [it["image_id"] for it in items],
        enc.encode_texts([it["caption"] for it in items]),
    )
    return index, enc


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    description = argv[0]
    index, enc = build_fixture_index()
    clues = parse_fallback(description)
    hyps = hypotheses(clues)
    print(f"[STUB ENCODER — plumbing demo, not retrieval quality]")
    print(f"description: {description!r} → {len(clues)} clue(s), {len(hyps)} branch(es)")
    branch_queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
    results = branch_union(index, branch_queries, per_branch_k=50)
    for rank, cand in enumerate(results[:5], start=1):
        print(f"  #{rank} {cand.image_id} score={cand.score:.4f} via {cand.branches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
