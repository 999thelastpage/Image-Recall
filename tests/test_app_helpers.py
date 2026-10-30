from pathlib import Path

from PIL import Image

from backend.app import auto_name, make_thumbnails
from backend.recall.manifest import ManifestEntry


def test_auto_name_never_blocks():
    assert auto_name("", set()) == "photos"
    assert auto_name("My Holiday 2024!", set()) == "My-Holiday-2024"
    assert auto_name("holiday", {"holiday"}) == "holiday-2"
    assert auto_name("holiday", {"holiday", "holiday-2"}) == "holiday-3"


def test_make_thumbnails_rewrites_paths_and_shrinks(tmp_path):
    src = tmp_path / "big.jpg"
    Image.new("RGB", (1200, 900), (200, 30, 30)).save(src, quality=95)
    entries = [ManifestEntry("img-000000", str(src), "x", src.stat().st_size, 0)]
    out = make_thumbnails(entries, "tcorp", tmp_path)
    thumb = Path(out[0].path)
    assert thumb.exists() and thumb != src
    assert thumb.stat().st_size < src.stat().st_size
    with Image.open(thumb) as im:
        assert max(im.size) <= 384
    # original entry fields preserved except path
    assert out[0].image_id == "img-000000" and out[0].row == 0
