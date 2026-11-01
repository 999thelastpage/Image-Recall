from PIL import Image

from backend.recall.evidence import claimed_count, dominant_colour, parse_boxes, parse_sam_box


def test_parse_boxes_counts_instances():
    txt = ("<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|><|mask;x/>"
           "<|box;x1=5;y1=6;x2=7;y2=8;w=10;h=10|><|mask;x/>")
    assert len(parse_boxes(txt)) == 2
    assert parse_boxes("") == []


def test_claimed_count():
    assert claimed_count("two dogs") == 2
    assert claimed_count("3 buses parked") == 3
    assert claimed_count("a red bus") == 1
    assert claimed_count("people eating") is None


def test_parse_sam_box():
    txt = "<0f>0<|box;x1=16;y1=16;x2=47;y2=47;w=64;h=64|><|mask;..."
    assert parse_sam_box(txt) == {"x1": 16, "y1": 16, "x2": 47, "y2": 47, "w": 64, "h": 64}
    assert parse_sam_box("no box here") is None
    assert parse_sam_box("") is None


def test_dominant_colour_reads_box(tmp_path):
    img = Image.new("RGB", (64, 64), (30, 30, 200))       # blue field
    for y in range(16, 48):
        for x in range(16, 48):
            img.putpixel((x, y), (220, 40, 30))            # red square
    path = tmp_path / "t.png"
    img.save(path)
    out = dominant_colour(str(path), {"x1": 16, "y1": 16, "x2": 47, "y2": 47, "w": 64, "h": 64})
    assert out["colour"] == "red"
    assert out["box"] == [16, 16, 47, 47]


class _Vision:
    def __init__(self, reply):
        self.reply = reply

    def chat_with_image(self, path, prompt, max_tokens=800):
        return self.reply


def test_muse_evidence_reads_found_count_and_colour():
    from backend.recall.evidence import muse_evidence

    got = muse_evidence(_Vision('x {"found": true, "count": 2, "colour": "Green"}'),
                        "p.jpg", "bus", said_colour="purple bus")
    assert got == {"found": True, "evidence_source": "Muse Spark",
                   "count_found": 2, "colour": "green"}


def test_muse_evidence_not_found_reports_no_count_or_colour():
    from backend.recall.evidence import muse_evidence

    got = muse_evidence(_Vision('{"found": false, "count": 3, "colour": "red"}'),
                        "p.jpg", "bus", said_colour="red bus")
    assert got == {"found": False, "evidence_source": "Muse Spark", "count_found": None}


def test_muse_evidence_raises_without_json():
    import pytest

    from backend.recall.evidence import muse_evidence

    with pytest.raises(ValueError):
        muse_evidence(_Vision("no answer"), "p.jpg", "bus")
