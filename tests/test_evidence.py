from PIL import Image

from backend.recall.evidence import dominant_colour, parse_sam_box


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
