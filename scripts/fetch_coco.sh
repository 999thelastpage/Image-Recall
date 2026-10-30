#!/usr/bin/env bash
# Fetch COCO val2017 images + captions into data/coco/ (git-ignored).
# Source: official COCO download URLs.
set -euo pipefail
mkdir -p data/coco
cd data/coco
[ -f val2017.zip ] || curl -O http://images.cocodataset.org/zips/val2017.zip
[ -f annotations_trainval2017.zip ] || curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q -o val2017.zip
unzip -q -o annotations_trainval2017.zip annotations/captions_val2017.json
python3 - <<'PY'
import json
from pathlib import Path
ann = json.loads(Path("annotations/captions_val2017.json").read_text())
name_by_id = {im["id"]: im["file_name"] for im in ann["images"]}
caps: dict[str, str] = {}
for a in ann["annotations"]:
    caps.setdefault(name_by_id[a["image_id"]], a["caption"])
Path("captions.json").write_text(json.dumps(caps))
print(f"{len(caps)} captioned val images")
PY
