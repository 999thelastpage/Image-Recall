"""Ingest a directory of images: manifest (checksums + row mapping),
thumbnails, and MetaCLIP embeddings aligned to manifest rows.

    .venv/bin/python -m scripts.ingest --images data/coco/val2017 --name coco_val2017

Outputs (data/ and embeddings/ are git-ignored except manifests):
    data/manifests/<name>.jsonl   one JSON object per image
    data/thumbs/<name>/...        256px thumbnails (optional)
    embeddings/<name>.npy         float32, row i = manifest row i
    embeddings/<name>.meta.json   model, dim, count, date
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
    ap.add_argument("--images", default="data/coco/val2017")
    ap.add_argument("--name", default="coco_val2017")
    ap.add_argument("--thumbs", action="store_true", help="also write 256px thumbnails")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    from backend.recall.config import ENCODER_MODEL
    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import build_manifest, write_manifest

    img_dir = Path(args.images)
    entries = build_manifest(img_dir)
    if not entries:
        print(f"no images under {img_dir}")
        return 1
    print(f"manifest: {len(entries)} images")

    if args.thumbs:
        thumb_dir = Path("data/thumbs") / args.name
        thumb_dir.mkdir(parents=True, exist_ok=True)
        for e in entries:
            out = thumb_dir / (Path(e.path).stem + ".jpg")
            if not out.exists():
                im = Image.open(e.path).convert("RGB")
                im.thumbnail((256, 256))
                im.save(out, quality=80)
        print(f"thumbs: {thumb_dir}")

    enc = MetaClipEncoder()
    print(f"encoder: {ENCODER_MODEL} on {enc.device}")
    t0 = time.time()
    mat = enc.encode_images([e.path for e in entries])
    dt = time.time() - t0
    print(f"embedded {len(entries)} in {dt:.1f}s = {len(entries) / dt:.1f} img/s")

    Path("embeddings").mkdir(exist_ok=True)
    np.save(f"embeddings/{args.name}.npy", mat)
    meta = {
        "name": args.name, "model": ENCODER_MODEL, "dim": int(mat.shape[1]),
        "count": int(mat.shape[0]), "device": enc.device,
        "images_per_sec": round(len(entries) / dt, 1),
    }
    Path(f"embeddings/{args.name}.meta.json").write_text(json.dumps(meta, indent=1))
    write_manifest(entries, Path(f"data/manifests/{args.name}.jsonl"))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
