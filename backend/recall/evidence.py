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


def muse_evidence(client, image_path: str, subject: str,
                  said_colour: str | None = None) -> dict:
    """Muse vision reads the same facts when SAM is not available.
    Raises if the reply has no usable JSON; the caller reports that."""
    import json

    ask = (f'Look at this photo. Is "{subject}" visible? Return ONLY JSON: '
           '{"found": true|false, "count": <integer or null>'
           + (', "colour": "<main colour of it, one word, or null>"' if said_colour else "")
           + "}")
    txt = client.chat_with_image(image_path, ask)
    data = json.loads(txt[txt.index("{") : txt.rindex("}") + 1])
    found = data.get("found") is True
    count = data.get("count")
    out: dict = {"found": found, "evidence_source": "Muse Spark",
                 "count_found": count if found and isinstance(count, int) and count > 0 else None}
    colour = data.get("colour")
    if found and said_colour and isinstance(colour, str) and colour.strip():
        out["colour"] = colour.strip().lower()
    return out


def parse_sam_box(text: str) -> dict | None:
    boxes = parse_boxes(text)
    return boxes[0] if boxes else None


def parse_boxes(text: str) -> list[dict]:
    """One box per segmented instance — len() is SAM's count."""
    return [
        {"x1": int(a), "y1": int(b), "x2": int(c), "y2": int(d),
         "w": int(w), "h": int(h)}
        for a, b, c, d, w, h in _BOX.findall(text or "")
    ]


_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "pair": 2,
}


def claimed_count(text: str) -> int | None:
    """'two dogs' -> 2, '3 buses' -> 3, 'a dog' -> 1, else None."""
    import re

    m = re.search(r"\b(\d+)\b", text or "")
    if m:
        return int(m.group(1))
    for tok in re.findall(r"[a-z]+", (text or "").lower()):
        if tok in _NUMBERS:
            return _NUMBERS[tok]
    return None


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
