"""Promote a per-model embedding cache to the live default.

After a benchmark win, this copies embeddings/<corpus>__<model>.npy
over embeddings/<corpus>.npy and updates the meta file, so the
server (which loads the default name) serves the new encoder.
Restart the server afterwards. Old default is kept as .b32.bak.

    .venv/bin/python -m scripts.swap_encoder --model facebook/metaclip-l14-fullcc2.5b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpora", nargs="*", default=["coco_val2017", "train2014"])
    args = ap.parse_args()
    slug = args.model.replace("/", "_")
    for corpus in args.corpora:
        src = Path(f"embeddings/{corpus}__{slug}.npy")
        dst = Path(f"embeddings/{corpus}.npy")
        meta_path = Path(f"embeddings/{corpus}.meta.json")
        if not src.exists():
            print(f"skip {corpus}: no cache {src}")
            continue
        if dst.exists():
            shutil.copy2(dst, Path(f"embeddings/{corpus}.b32.bak.npy"))
        shutil.copy2(src, dst)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"name": corpus}
        import numpy as np

        meta.update({"model": args.model, "dim": int(np.load(dst, mmap_mode="r").shape[1])})
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{corpus}: now served by {args.model} (dim {meta['dim']})")
    print("restart the server to pick up the new encoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
