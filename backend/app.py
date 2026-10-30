"""Recall local test server (FastAPI).

Serves the static UI and three job families:
  corpora  - list ingested corpora (manifest + embeddings on disk)
  ingest   - point at ANY local folder; background job embeds every
             image (MetaCLIP, GPU when available) with progress
  search   - natural-language query over a corpus: plain baseline,
             or Recall (Muse clue chips + leave-one-clue-out union)

Local development tool: binds localhost by default. Folder paths
are read from this machine only; nothing is uploaded anywhere.
Muse parsing is used only when META_MODEL_API_KEY is set (.env).
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.recall.clues import Clue, hypotheses, parse_fallback
from backend.recall.embeddings import MetaClipEncoder
from backend.recall.manifest import build_manifest, read_manifest, write_manifest
from backend.recall.providers import MuseClient, ProviderDisabled, load_dotenv
from backend.recall.rank import branch_union
from backend.recall.retrieve import ExactIndex

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(str(ROOT / ".env"))

app = FastAPI(title="Recall — local test UI")

_encoder: MetaClipEncoder | None = None
_encoder_lock = threading.Lock()
_indexes: dict[str, ExactIndex] = {}
_jobs: dict[str, dict] = {}


def get_encoder() -> MetaClipEncoder:
    global _encoder
    with _encoder_lock:
        if _encoder is None:
            _encoder = MetaClipEncoder()
        return _encoder


def corpus_paths(name: str) -> tuple[Path, Path]:
    return ROOT / "data/manifests" / f"{name}.jsonl", ROOT / "embeddings" / f"{name}.npy"


def get_index(name: str) -> ExactIndex:
    if name not in _indexes:
        manifest_path, emb_path = corpus_paths(name)
        if not manifest_path.exists() or not emb_path.exists():
            raise HTTPException(404, f"corpus '{name}' not ingested")
        entries = read_manifest(manifest_path)
        _indexes[name] = ExactIndex([e.image_id for e in entries], np.load(emb_path))
        _indexes[name]._entries = entries  # type: ignore[attr-defined]
    return _indexes[name]


@app.get("/api/corpora")
def corpora() -> list[dict]:
    out = []
    for meta in sorted((ROOT / "embeddings").glob("*.meta.json")):
        import json

        info = json.loads(meta.read_text())
        out.append(info)
    return out


def _clean_name(raw: str) -> str:
    name = "".join(c for c in raw if c.isalnum() or c in "-_").strip()
    if not name:
        raise HTTPException(400, "give the corpus a name (letters/digits/-/_))")
    return name


def auto_name(raw: str, taken: set[str]) -> str:
    """Never blocks: sanitize, default to 'photos', de-duplicate."""
    base = "".join(c if (c.isalnum() or c in "-_") else "-" for c in (raw or "")).strip("-")
    base = "-".join(p for p in base.split("-") if p) or "photos"
    name, n = base, 2
    while name in taken:
        name = f"{base}-{n}"
        n += 1
    return name


def _taken_names() -> set[str]:
    return {p.stem.removesuffix(".meta") for p in (ROOT / "embeddings").glob("*.meta.json")}


def make_thumbnails(entries, name: str, root: Path):
    """Write <=384px JPEGs under data/thumbs/<name>/ and return
    manifest entries pointing at them (uploads keep no full copies)."""
    from PIL import Image

    from backend.recall.manifest import ManifestEntry

    thumb_dir = root / "data/thumbs" / name
    thumb_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for e in entries:
        dest = thumb_dir / (Path(e.path).stem + ".jpg")
        im = Image.open(e.path).convert("RGB")
        im.thumbnail((384, 384))
        im.save(dest, quality=80)
        out.append(ManifestEntry(e.image_id, str(dest), e.sha256, dest.stat().st_size, e.row))
    return out


def _new_job(name: str, folder: Path) -> dict:
    job_id = uuid.uuid4().hex[:8]
    job = {"id": job_id, "name": name, "folder": str(folder), "status": "queued",
           "done": 0, "total": 0, "images_per_sec": None, "error": None}
    _jobs[job_id] = job
    return job


def _spawn_ingest(job: dict, folder: Path, name: str, shrink_uploads: bool = False) -> dict:
    def run() -> None:
        try:
            job["status"] = "scanning"
            entries = build_manifest(folder)
            job["total"] = len(entries)
            if not entries:
                raise ValueError("no images (.jpg/.jpeg/.png/.webp/.bmp) in folder")
            job["status"] = "embedding"
            enc = get_encoder()
            paths = [e.path for e in entries]
            t0 = time.time()
            chunks = []
            for i in range(0, len(paths), 256):
                chunks.append(enc.encode_images(paths[i : i + 256]))
                job["done"] = min(i + 256, len(paths))
            mat = np.concatenate(chunks)
            job["images_per_sec"] = round(len(paths) / (time.time() - t0), 1)
            (ROOT / "embeddings").mkdir(exist_ok=True)
            np.save(ROOT / "embeddings" / f"{name}.npy", mat)
            import json

            (ROOT / "embeddings" / f"{name}.meta.json").write_text(json.dumps({
                "name": name, "model": "facebook/metaclip-b32-fullcc2.5b",
                "dim": int(mat.shape[1]), "count": int(mat.shape[0]),
                "device": enc.device, "images_per_sec": job["images_per_sec"],
                "folder": str(folder)}))
            if shrink_uploads:
                entries = make_thumbnails(entries, name, ROOT)
                uploads_root = (ROOT / "data/uploads").resolve()
                if uploads_root in folder.resolve().parents:
                    for e in list(folder.iterdir()):
                        if e.is_file():
                            e.unlink()
                    job["kept_copies"] = "thumbnails only"
            manifest_path, _ = corpus_paths(name)
            write_manifest(entries, manifest_path)
            _indexes.pop(name, None)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=run, daemon=True).start()
    return job


class IngestRequest(BaseModel):
    folder: str
    name: str


@app.post("/api/ingest")
def start_ingest(req: IngestRequest) -> dict:
    folder = Path(req.folder).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"folder not found: {req.folder}")
    name = _clean_name(req.name)
    return _spawn_ingest(_new_job(name, folder), folder, name)


@app.post("/api/upload-ingest")
async def upload_ingest(
    name: str = Form(""), keep_originals: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> dict:
    """Folder picked in the browser: photos are sent to this local
    server (127.0.0.1 only — nothing leaves the machine), embedded,
    and by default only <=384px thumbnails are kept afterwards
    (embeddings ~2 KB/photo). Set keep_originals to retain copies."""
    from backend.recall.manifest import IMAGE_SUFFIXES

    name = auto_name(name, _taken_names())
    dest = ROOT / "data/uploads" / name
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        raw = Path(f.filename or "").name  # basename only: no path traversal
        if not raw or Path(raw).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = dest / raw
        n = 1
        while target.exists():
            target = dest / f"{Path(raw).stem}_{n}{Path(raw).suffix}"
            n += 1
        target.write_bytes(await f.read())
        saved += 1
    if not saved:
        raise HTTPException(400, "no image files in that folder")
    return _spawn_ingest(_new_job(name, dest), dest, name,
                         shrink_uploads=not keep_originals)


@app.get("/api/ingest/{job_id}")
def ingest_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


class SearchRequest(BaseModel):
    query: str
    corpus: str = "coco_val2017"
    mode: str = "both"  # baseline | recall | both
    top_k: int = 12


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    if req.mode not in ("baseline", "recall", "both"):
        raise HTTPException(400, "mode must be baseline, recall or both")
    index = get_index(req.corpus)
    entries = {e.image_id: e for e in index._entries}  # type: ignore[attr-defined]
    enc = get_encoder()
    timings: dict[str, float] = {}

    def item(image_id: str, score: float, branches: list[str] | None = None) -> dict:
        e = entries[image_id]
        return {"image_id": image_id, "score": round(score, 4),
                "file": Path(e.path).name, "branches": branches or [],
                "url": f"/api/image/{req.corpus}/{image_id}"}

    out: dict = {"corpus": req.corpus, "corpus_size": len(index), "query": req.query}

    if req.mode in ("baseline", "both"):
        t0 = time.time()
        q = enc.encode_texts([req.query])
        timings["encode_query_ms"] = round((time.time() - t0) * 1000, 1)
        t0 = time.time()
        hits = index.search(q, req.top_k)
        timings["baseline_search_ms"] = round((time.time() - t0) * 1000, 1)
        out["baseline"] = [item(i, s) for i, s in hits]

    if req.mode in ("recall", "both"):
        t0 = time.time()
        parse_source = "fallback"
        try:
            chips = MuseClient().parse(req.query)
            clues = [Clue(c["text"]) for c in chips
                     if len(c["text"].strip()) > 1]  # drop filler chips
            parse_source = "Muse Spark"
        except (ProviderDisabled, Exception):
            clues = parse_fallback(req.query)
        timings["parse_ms"] = round((time.time() - t0) * 1000, 1)
        hyps = hypotheses(clues)
        t0 = time.time()
        queries = [(h, enc.encode_texts([h.query_text])) for h in hyps]
        union = branch_union(index, queries, per_branch_k=50)
        timings["recall_search_ms"] = round((time.time() - t0) * 1000, 1)
        out["clues"] = [c.text for c in clues]
        out["parse_source"] = parse_source
        out["branches"] = [h.label for h in hyps]
        out["union_size"] = len(union)
        out["recall"] = [item(c.image_id, c.score, c.branches)
                         for c in union[: req.top_k]]

    out["timings"] = timings
    return out


@app.get("/api/image/{corpus}/{image_id}")
def image(corpus: str, image_id: str):
    index = get_index(corpus)
    for e in index._entries:  # type: ignore[attr-defined]
        if e.image_id == image_id:
            if not Path(e.path).exists():
                raise HTTPException(404, "image file moved/deleted")
            return FileResponse(e.path)
    raise HTTPException(404, "unknown image")


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="ui")
