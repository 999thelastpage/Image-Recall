"""SAM-grounded evidence: segment the referred object, then read
its dominant colour inside the returned box.

The mask payload in SAM's output uses a compact encoding; the box
coordinates are plain text and reliable. We sample the box's inner
region (shrunk 15% to avoid edges) and name the nearest colour.
Labelled honestly as in-box colour wherever shown.
"""

from __future__ import annotations

import re

_BOX = re.compile(r"x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)")

COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (200, 40, 35), "orange": (235, 130, 30), "yellow": (235, 205, 40),
    "green": (60, 160, 65), "blue": (45, 90, 200), "purple": (130, 70, 170),
    "pink": (235, 150, 180), "brown": (130, 85, 50), "black": (25, 25, 25),
    "white": (240, 240, 240), "gray": (140, 140, 140),
}


def parse_sam_box(text: str) -> dict | None:
    m = _BOX.search(text or "")
    if not m:
        return None
    x1, y1, x2, y2, w, h = (int(g) for g in m.groups())
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h}


def dominant_colour(image_path: str, box: dict) -> dict:
    """Nearest named colour of the mean RGB inside the (scaled) box."""
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        sx, sy = im.width / max(box["w"], 1), im.height / max(box["h"], 1)
        x1, y1 = int(box["x1"] * sx), int(box["y1"] * sy)
        x2, y2 = int(box["x2"] * sx), int(box["y2"] * sy)
        # shrink 15% inward to sample the object, not its border
        dx, dy = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        crop = im.crop((x1 + dx, y1 + dy, max(x1 + dx + 1, x2 - dx),
                        max(y1 + dy + 1, y2 - dy))).resize((16, 16))
        px = list(crop.getdata())
    n = len(px)
    mean = tuple(sum(p[i] for p in px) // n for i in range(3))
    name = min(COLOURS, key=lambda c: sum((a - b) ** 2 for a, b in
                                          zip(mean, COLOURS[c], strict=True)))
    return {"colour": name, "rgb": list(mean),
            "box": [x1, y1, x2, y2]}
