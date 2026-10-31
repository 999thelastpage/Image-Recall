from pathlib import Path

from PIL import Image

import backend.app as app_module
from backend.app import (
    _doubt_aware,
    _parse_cache_get,
    _parse_cache_put,
    auto_name,
    make_thumbnails,
    windows_to_wsl,
)
from backend.recall.clues import Clue
from backend.recall.embeddings import StubEncoder
from backend.recall.manifest import ManifestEntry
from backend.recall.questions import NOT_SURE, SOMETHING_ELSE
from backend.recall.retrieve import ExactIndex
from backend.recall.session import SearchSession


def test_windows_paths_translate_to_wsl():
    assert windows_to_wsl(r"E:\GenerativeAI\recall-image\train2014") == \
        "/mnt/e/GenerativeAI/recall-image/train2014"
    assert windows_to_wsl("C:/Users/Kuro/Pictures") == "/mnt/c/Users/Kuro/Pictures"
    # Linux paths pass through untouched
    assert windows_to_wsl("/home/you/photos") == "/home/you/photos"
    assert windows_to_wsl("relative/dir") == "relative/dir"


def test_parse_cache_normalises_and_copies():
    chips = [{"text": "red bus", "kind": "colour", "referent": "bus"}]
    _parse_cache_put("A  Red   BUS on a street", chips)
    # case/whitespace variants hit the same entry
    hit = _parse_cache_get("a red bus on a street")
    assert hit == chips
    # callers get copies: mutating a hit must not poison the cache
    hit[0]["text"] = "mutated"
    assert _parse_cache_get("a red bus on a street")[0]["text"] == "red bus"
    assert _parse_cache_get("some other query") is None


def test_demo_mode_blocks_writes(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setattr(app_module, "DEMO_MODE", True)
    assert app_module.meta() == {"demo": True}
    for call in (
        lambda: app_module.start_ingest(
            app_module.IngestRequest(folder="/tmp", name="x")),
        lambda: asyncio.run(app_module.upload_ingest(
            name="x", keep_originals=False, files=[])),
    ):
        try:
            call()
            raise AssertionError("expected 403 in demo mode")
        except HTTPException as e:
            assert e.status_code == 403
    monkeypatch.setattr(app_module, "DEMO_MODE", False)
    assert app_module.meta() == {"demo": False}


def test_doubt_aware_bundled_anchor_doubts_only_subject_and_passes_category():
    enc = StubEncoder()
    captions = [
        "purple bus parked in grass",
        "purple car parked on street",
        "red car parked in grass",
        "blue bus parked in grass",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("purple car", "colour", "car"),
             Clue("in grass", "setting", "grass")]
    s = SearchSession(index, enc, "a purple car in grass", clues)
    s.start()
    assert s.pending is not None and s.pending.kind == "subject"
    s.answer(NOT_SURE)
    assert s.pending is not None and s.pending.kind == "category"
    s.answer("A vehicle")
    out = _doubt_aware(s)
    assert "the subject 'car'" in out
    assert "wrong about: purple car" not in out
    assert "confirmed the subject was a vehicle" in out


def _dogs_app_session() -> SearchSession:
    enc = StubEncoder()
    captions = [
        "two dogs; together",
        "two cats; together",
        "one dog; alone",
        "two birds; together",
    ]
    ids = [f"img-{i:06d}" for i in range(len(captions))]
    index = ExactIndex(ids, enc.encode_texts(captions))
    clues = [Clue("two dogs", "count", "dogs"),
             Clue("together", "other", "dogs")]
    s = SearchSession(index, enc, "two dogs together", clues)
    s.start()
    return s


def test_doubt_aware_subject_not_sure_scores_dog_photos_low():
    s = _dogs_app_session()
    assert s.pending is not None and s.pending.kind == "subject"
    s.answer(NOT_SURE)
    assert 0 in s.dropped
    out = _doubt_aware(s)
    assert "probably not the one" in out
    assert "score it low" in out
    assert (
        "The user doubted the subject was 'dogs'; a photo whose main "
        "subject is dogs is probably not the one: score it low even "
        "if the other details match."
    ) in out


def test_doubt_aware_replaced_subject_not_excluded():
    s = _dogs_app_session()
    s.answer(SOMETHING_ELSE)
    assert s.dropped == set()
    s.answer("cat")  # replacement: anchor NOT dropped
    assert s.dropped == set()
    out = _doubt_aware(s)
    assert "probably not the one" not in out
    assert "score it low" not in out


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
