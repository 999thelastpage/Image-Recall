"""Prepare the hosted-demo data for coco_val2017.

The hosted demo cannot serve the COCO originals (788 MB, local
paths), so it serves <=384px thumbnails instead and a demo manifest
that points at them. Embeddings are untouched: rows already align
with the source manifest, and thumbnails keep the same row order.

    .venv/bin/python -m scripts.prep_demo

Outputs:
    data/thumbs/coco_val2017/<stem>.jpg        (git-ignored)
    data/manifests/coco_val2017.demo.jsonl     demo-only manifest
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
NAME = "coco_val2017"
MAX_PX = 384


def main() -> int:
    from PIL import Image

    from backend.recall.manifest import ManifestEntry, read_manifest, write_manifest

    src = ROOT / "data/manifests" / f"{NAME}.jsonl"
    entries = read_manifest(src)
    if not entries:
        print(f"no entries in {src}")
        return 1
    thumb_dir = ROOT / "data/thumbs" / NAME
    thumb_dir.mkdir(parents=True, exist_ok=True)
    out: list[ManifestEntry] = []
    made = 0
    for e in entries:
        dest = thumb_dir / (Path(e.path).stem + ".jpg")
        if not dest.exists():
            with Image.open(ROOT / e.path) as im:
                im = im.convert("RGB")
                im.thumbnail((MAX_PX, MAX_PX))
                im.save(dest, quality=80)
            made += 1
        rel = f"data/thumbs/{NAME}/{dest.name}"
        out.append(ManifestEntry(e.image_id, rel, e.sha256,
                                 dest.stat().st_size, e.row))
    demo_manifest = ROOT / "data/manifests" / f"{NAME}.demo.jsonl"
    write_manifest(out, demo_manifest)
    total_mb = sum(e.size_bytes for e in out) / 1e6
    print(f"thumbs: {len(out)} ({made} new) in {thumb_dir}, {total_mb:.0f} MB")
    print(f"demo manifest: {demo_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
