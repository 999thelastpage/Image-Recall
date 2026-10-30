"""Deterministic 200-item fixture for plumbing tests.

Writes data/fixture_captions.json with synthetic captions. Used
with StubEncoder to exercise ingest → search end to end. It makes
NO retrieval-quality claim.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

COLOURS = ["red", "orange", "blue", "green", "black", "white"]
OBJECTS = ["jacket", "bicycle", "dog", "car", "chair", "boat"]
SETTINGS = ["beach", "park", "street", "lake", "market", "garden"]


def main() -> None:
    rng = random.Random(42)
    items = []
    for i in range(200):
        caption = (
            f"{rng.choice(COLOURS)} {rng.choice(OBJECTS)}; "
            f"{rng.choice(SETTINGS)}; daylight"
        )
        items.append({"image_id": f"img-{i:06d}", "caption": caption})
    out = Path("data/fixture_captions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"wrote {len(items)} fixture items to {out}")


if __name__ == "__main__":
    main()
