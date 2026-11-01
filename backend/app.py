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

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.recall.clues import Clue, dedupe_clues, parse_fallback, recall_hypotheses
from backend.recall.embeddings import MetaClipEncoder
from backend.recall.manifest import build_manifest, read_manifest, write_manifest
from backend.recall.providers import MuseClient, load_dotenv, rerank_client
from backend.recall.config import ENCODER_MODEL, FINAL_TOP_K, branch_quota
from backend.recall.questions import COLOURS
from backend.recall.rank import calibrated_union
from backend.recall.retrieve import ExactIndex

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(str(ROOT / ".env"))

# Hosted-demo mode (RECALL_DEMO=1, set by deploy/modal_app.py):
# anyone on the internet can search, ask questions, rerank and view
# evidence: but nobody can write to the container's disk through
# the folder-ingest or upload endpoints.
DEMO_MODE = os.environ.get("RECALL_DEMO") == "1"


def _demo_guard() -> None:
    if DEMO_MODE:
        raise HTTPException(
            403, "adding photo folders is disabled in the hosted demo")

app = FastAPI(title="Recall: local test UI")


@app.get("/api/meta")
def meta() -> dict:
    return {"demo": DEMO_MODE}

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


# --- Muse clue-parse cache -------------------------------------
# A Muse parse is a hosted reasoning-model call (measured 7–16 s);
# the local lookup after it is ~0.1 s. Identical or whitespace/case-
# variant repeats of a query must not pay for the API again.
_parse_cache: dict[str, list[dict]] = {}
_parse_cache_lock = threading.Lock()
_PARSE_CACHE_MAX = 256


def _parse_cache_key(query: str) -> str:
    return " ".join((query or "").lower().split())


def _parse_cache_get(query: str) -> list[dict] | None:
    with _parse_cache_lock:
        hit = _parse_cache.get(_parse_cache_key(query))
        return [dict(c) for c in hit] if hit is not None else None


def _parse_cache_put(query: str, chips: list[dict]) -> None:
    with _parse_cache_lock:
        if len(_parse_cache) >= _PARSE_CACHE_MAX:
            _parse_cache.pop(next(iter(_parse_cache)))  # evict oldest
        _parse_cache[_parse_cache_key(query)] = [dict(c) for c in chips]


_parse_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="muse-parse")


def _parse_chips(query: str) -> tuple[list[dict] | None, str, bool]:
    """(chips, source, cached). chips is None when Muse is
    unavailable or failed: callers then use the offline fallback."""
    hit = _parse_cache_get(query)
    if hit is not None:
        return hit, "Muse Spark", True
    try:
        chips = MuseClient().parse(query)
    except Exception:  # noqa: BLE001: disabled/key/network/parse
        return None, "fallback", False
    usable = [c for c in chips if len(str(c.get("text", "")).strip()) > 1]
    if usable:
        _parse_cache_put(query, usable)
    return usable, "Muse Spark", False


@app.get("/api/corpora")
def corpora() -> list[dict]:
    out = []
    for meta in sorted((ROOT / "embeddings").glob("*.meta.json")):
        import json

        info = json.loads(meta.read_text())
        out.append(info)
    return out


def windows_to_wsl(p: str) -> str:
    """'E:\\photos\\a' -> '/mnt/e/photos/a'. Linux paths pass through.
    Lets users paste Windows paths from Explorer into the UI."""
    import re

    m = re.match(r"^([A-Za-z]):[\\/](.*)$", (p or "").strip())
    if not m:
        return p
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


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
                "name": name, "model": ENCODER_MODEL,
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
    _demo_guard()
    folder = Path(windows_to_wsl(req.folder)).expanduser()
    if not folder.is_dir():
        raise HTTPException(
            400,
            f"folder not found: {folder}"
            + (f" (translated from {req.folder!r})" if str(folder) != req.folder else ""),
        )
    name = auto_name(req.name or folder.name, _taken_names())
    return _spawn_ingest(_new_job(name, folder), folder, name)


@app.post("/api/upload-ingest")
async def upload_ingest(
    name: str = Form(""), keep_originals: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> dict:
    """Folder picked in the browser: photos are sent to this local
    server (127.0.0.1 only: nothing leaves the machine), embedded,
    and by default only <=384px thumbnails are kept afterwards
    (embeddings ~2 KB/photo). Set keep_originals to retain copies."""
    _demo_guard()
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

    # Start the hosted Muse parse concurrently: baseline (and the
    # UI's first results) never wait on the network for it.
    parse_future = None
    if req.mode in ("recall", "both"):
        parse_future = _parse_pool.submit(_parse_chips, req.query)

    if req.mode in ("baseline", "both"):
        t0 = time.time()
        q = enc.encode_texts([req.query])
        timings["encode_query_ms"] = round((time.time() - t0) * 1000, 1)
        t0 = time.time()
        hits = index.search(q, req.top_k)
        timings["baseline_search_ms"] = round((time.time() - t0) * 1000, 1)
        out["baseline"] = [item(i, s) for i, s in hits]

    if parse_future is not None:
        t0 = time.time()
        chips, parse_source, cached = parse_future.result()
        if chips is None:
            clues = dedupe_clues(parse_fallback(req.query))
        else:
            clues = dedupe_clues([Clue(c["text"], c.get("kind", "other"),
                                       c.get("referent", ""))
                                  for c in chips])
        timings["parse_ms"] = round((time.time() - t0) * 1000, 1)
        out["parse_cached"] = cached
        hyps = recall_hypotheses(req.query, clues,
                                 substitution_colours=tuple(COLOURS),
                                 max_substitutions=2)
        t0 = time.time()
        # Batched encode + batched union: one matmul for all branches.
        mat = enc.encode_texts([h.query_text for h in hyps]) if hyps else []
        queries = list(zip(hyps, mat, strict=True))
        union, _weights = calibrated_union(index, queries, branch_quota(len(index)))
        timings["recall_search_ms"] = round((time.time() - t0) * 1000, 1)
        out["clues"] = [c.text for c in clues]
        out["parse_source"] = parse_source
        out["branches"] = [h.label for h in hyps]
        out["union_size"] = len(union)
        out["recall"] = [item(c.image_id, c.score, c.branches)
                         for c in union[: req.top_k]]

    out["timings"] = timings
    return out


_sessions: dict[str, dict] = {}

# Paid AI checks: repeats are free, and new checks have an hourly limit.
RERANKS_PER_HOUR = int(os.environ.get("RERANKS_PER_HOUR", "30"))
_rerank_scores: dict = {}
_rerank_times: list[float] = []


def _take_rerank_slot(now: float | None = None) -> bool:
    """False when the hourly limit is used. Does not persist across restarts."""
    now = time.time() if now is None else now
    _rerank_times[:] = [t for t in _rerank_times if now - t < 3600]
    if len(_rerank_times) >= RERANKS_PER_HOUR:
        return False
    _rerank_times.append(now)
    return True


def _new_reranker(client, max_workers: int = 8):
    from backend.recall.rerank import MuseReranker

    if len(_rerank_scores) > 20000:
        _rerank_scores.clear()
    return MuseReranker(client, max_workers=max_workers, cache=_rerank_scores)


PRIMARY_RETRY_SECONDS = 600
_primary_retry_at = [0.0]


def _ai_check(candidates: list, paths: dict, description: str) -> tuple[list, str]:
    """(ranked, model name). Tries the main model, then the backup. After the
    main model fails, it waits before the next try, so a search stays fast."""
    from backend.recall.providers import BackupReranker
    from backend.recall.rerank import RERANK_TOP_N, MuseReranker, ScoreTable

    if time.time() >= _primary_retry_at[0] and _take_rerank_slot():
        try:
            client = rerank_client()
            ranked = _new_reranker(client).rerank(candidates, paths, description)
            if any(r.muse is not None for r in ranked):
                return ranked, client.model
        except Exception:  # noqa: BLE001: the backup is next
            pass
        _primary_retry_at[0] = time.time() + PRIMARY_RETRY_SECONDS
    backup = BackupReranker()  # raises when not configured
    head = [paths[c.image_id] for c in candidates[:RERANK_TOP_N]]
    logits = backup.logits(description, head)
    if not logits:
        raise RuntimeError("no image could be scored")
    table = ScoreTable(backup.model, logits)
    return MuseReranker(table, max_workers=8).rerank(candidates, paths, description), backup.model



def _save_session(rec: dict) -> None:
    import json

    d = ROOT / "data/sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec['session'].session_id}.json").write_text(json.dumps(
        {"corpus": rec["corpus"], "session": rec["session"].to_dict()}))


def _get_rec(session_id: str) -> dict:
    rec = _sessions.get(session_id)
    if rec is not None:
        return rec
    path = ROOT / "data/sessions" / f"{session_id}.json"
    if path.exists():
        import json

        from backend.recall.session import SearchSession

        payload = json.loads(path.read_text())
        index = get_index(payload["corpus"])
        s = SearchSession.from_dict(index, get_encoder(), payload["session"])
        rec = {"session": s, "corpus": payload["corpus"], "index": index}
        _sessions[session_id] = rec
        return rec
    raise HTTPException(404, "unknown session")


class SessionRequest(BaseModel):
    query: str
    corpus: str = "coco_val2017"


def _enrich(corpus: str, index, snap: dict) -> dict:
    entries = {e.image_id: e for e in index._entries}  # type: ignore[attr-defined]
    for r in snap.get("results", []):
        e = entries.get(r["image_id"])
        r["url"] = f"/api/image/{corpus}/{r['image_id']}"
        r["file"] = Path(e.path).name if e else ""
    return snap


@app.post("/api/session")
def create_session(req: SessionRequest) -> dict:
    index = get_index(req.corpus)
    enc = get_encoder()
    chips, source, _cached = _parse_chips(req.query)
    if chips is None:
        clues = dedupe_clues(parse_fallback(req.query))
    else:
        clues = dedupe_clues([Clue(c["text"], c.get("kind", "other"),
                                   c.get("referent", ""))
                              for c in chips])
    if not clues:
        raise HTTPException(400, "could not parse any clues from that description")
    from backend.recall.session import SearchSession

    s = SearchSession(index, enc, req.query, clues)
    s.start()
    _sessions[s.session_id] = {"session": s, "corpus": req.corpus, "index": index}
    _save_session(_sessions[s.session_id])
    snap = _enrich(req.corpus, index, s.snapshot())
    snap["parse_source"] = source
    return snap


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    rec = _get_rec(session_id)
    return _enrich(rec["corpus"], rec["index"], rec["session"].snapshot())


class AnswerRequest(BaseModel):
    value: str


@app.post("/api/session/{session_id}/answer")
def answer_session(session_id: str, req: AnswerRequest) -> dict:
    rec = _get_rec(session_id)
    try:
        rec["session"].answer(req.value)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    _save_session(rec)
    return _enrich(rec["corpus"], rec["index"], rec["session"].snapshot())


def _doubt_aware(s) -> str:
    from backend.recall.questions import NOT_SURE
    from backend.recall.rerank import doubt_aware_description

    anchor_fn = getattr(s, "_anchor_index", None)
    anchor = anchor_fn() if callable(anchor_fn) else None
    dropped = []
    for i in sorted(s.dropped):
        clue = s.clues[i]
        referent = (getattr(clue, "referent", "") or "").strip()
        if anchor is not None and i == anchor and referent:
            dropped.append(f"the subject '{referent}'")
        else:
            dropped.append(clue.text)
    corrections = [
        (s.clues[a.clue_index].text, a.value)
        for a in s.answers
        if a.kind == "attribute" and a.clue_index is not None
        and a.value != NOT_SURE
    ]
    notes = [
        f"The user confirmed the subject was {a.value.lower()}."
        for a in s.answers
        if a.kind == "category" and a.value != NOT_SURE
    ]
    if anchor is not None and anchor in s.dropped:
        clue = s.clues[anchor]
        referent = (getattr(clue, "referent", "") or "").strip()
        if not referent:
            words = clue.text.split()
            referent = words[-1] if words else clue.text
        notes.append(
            f"The user doubted the subject was '{referent}'; a photo "
            f"whose main subject is {referent} is probably not the one: "
            f"score it low even if the other details match."
        )
    return doubt_aware_description(s.description, dropped, corrections,
                                   notes=notes)


@app.post("/api/session/{session_id}/finish")
def finish_session(session_id: str) -> dict:
    rec = _get_rec(session_id)
    s = rec["session"]
    out = s.finish()
    _save_session(rec)
    entries = {e.image_id: e for e in rec["index"]._entries}  # type: ignore[attr-defined]
    # Automatic doubt-aware rerank of the top 40: measured rescuing
    # failures the encoder order could not. Best-effort: without a
    # Muse key the encoder order stands and we say so.
    out["reranked"] = False
    try:
        paths = {i: e.path for i, e in entries.items()}
        ranked, out["rerank_model"] = _ai_check(s.candidates, paths, _doubt_aware(s))
        rec["final_ids"] = [r.image_id for r in ranked[:FINAL_TOP_K]]
        out["results"] = [
            {"image_id": r.image_id, "score": round(r.final, 4),
             "muse_score": (None if r.muse is None else round(r.muse, 2)),
             "branches": r.branches}
            for r in ranked[:FINAL_TOP_K]
        ]
        out["reranked"] = True
    except Exception as e:  # noqa: BLE001: surface, never silent
        out["rerank_error"] = f"{type(e).__name__}: rerank unavailable"
    for r in out["results"]:
        e = entries.get(r["image_id"])
        r["url"] = f"/api/image/{rec['corpus']}/{r['image_id']}"
        r["file"] = Path(e.path).name if e else ""
    return out


@app.post("/api/session/{session_id}/rerank")
def rerank_session(session_id: str) -> dict:
    rec = _get_rec(session_id)
    s = rec["session"]
    entries = {e.image_id: e for e in rec["index"]._entries}  # type: ignore[attr-defined]
    paths = {i: e.path for i, e in entries.items()}
    try:
        out, _model = _ai_check(s.candidates, paths, _doubt_aware(s))
        rec["final_ids"] = [r.image_id for r in out[:FINAL_TOP_K]]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"rerank unavailable: {type(e).__name__}") from e
    return {
        "session_id": session_id,
        "reranked": True,
        "results": [
            {"image_id": r.image_id, "score": round(r.final, 4),
             "encoder_score": round(r.encoder, 4),
             "muse_score": (None if r.muse is None else round(r.muse, 2)),
             "branches": r.branches,
             "url": f"/api/image/{rec['corpus']}/{r.image_id}",
             "file": Path(entries[r.image_id].path).name if r.image_id in entries else ""}
            for r in out[:12]
        ],
    }


@app.post("/api/session/{session_id}/evidence")
def session_evidence(session_id: str) -> dict:
    """Multi-dimension evidence for the session's top-3.

    Validates what the user claimed, dimension by dimension:
    presence + count (SAM instances), colour (in-box sampling,
    only when a colour was claimed), and setting/activity claims
    (Muse vision verdict per claim). Dimensions the user never
    claimed are not reported: no colour readings for people.
    """
    rec = _get_rec(session_id)
    s = rec["session"]
    entries = {e.image_id: e for e in rec["index"]._entries}  # type: ignore[attr-defined]
    from backend.recall.evidence import (
        claimed_count,
        dominant_colour,
        muse_evidence,
        parse_boxes,
    )

    colour = next((c for c in s.clues if c.kind == "colour"), None)
    count = next((c for c in s.clues if c.kind == "count"), None)
    anchor = next((c for c in s.clues if c.kind in ("object", "person")), None)
    prompt = (
        (colour.referent if colour and colour.referent else None)
        or (count.referent if count and count.referent else None)
        or (anchor.text if anchor else None)
        or (s.clues[0].text if s.clues else "the main subject")
    )
    said_colour = colour.text if colour else None
    said_count = claimed_count(count.text) if count else None
    if said_count is None and anchor is not None:
        said_count = claimed_count(anchor.text)  # 'two dogs' is one clue
    claims = [c.text for c in s.clues if c.kind in ("setting", "activity")]
    # Check what the user corrected to, not the words they first typed.
    from backend.recall.questions import NOT_SURE

    for a in s.answers:
        if a.kind != "attribute" or a.clue_index is None or a.value == NOT_SURE:
            continue
        clue = s.clues[a.clue_index]
        if clue is anchor or clue.kind in ("object", "person"):
            prompt, said_count = a.value, None
        elif clue is colour:
            said_colour = a.value
    try:
        from backend.recall.providers import MuseClient, SamClient

        muse = MuseClient()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"evidence unavailable: {type(e).__name__}") from e
    try:
        sam = SamClient()
    except Exception:  # noqa: BLE001: Muse vision reads the facts instead
        sam = None

    def check(cand) -> dict:
        e = entries.get(cand.image_id)
        item: dict = {"image_id": cand.image_id, "prompt": prompt,
                      "said": said_colour, "count_claimed": said_count}
        if e is None:
            item["error"] = "image unavailable"
            return item
        try:
            if sam is None:
                raise RuntimeError("SAM not available")
            got = sam.segment(e.path, prompt)
            boxes = parse_boxes(got["text"])
            item["seconds"] = got["seconds"]
            item["found"] = len(boxes) > 0
            item["count_found"] = len(boxes) if boxes else None
            item["evidence_source"] = "SAM 3.1"
            if boxes and said_colour:
                item.update(dominant_colour(e.path, boxes[0]))
        except Exception:  # noqa: BLE001: SAM down, Muse vision reads the facts
            try:
                item.update(muse_evidence(muse, e.path, prompt, said_colour))
            except Exception as ex:  # noqa: BLE001
                item["error"] = f"{type(ex).__name__}"
                return item
        if claims:
            try:
                ask = ("For each statement, answer yes, no, or unsure based "
                       "only on this photo. Return ONLY JSON like "
                       '{"verdicts": {"<statement>": "yes|no|unsure"}}.\n'
                       + "\n".join(f"- {c}" for c in claims))
                txt = muse.chat_with_image(e.path, ask)
                import json as _json

                data = _json.loads(txt[txt.index("{") : txt.rindex("}") + 1])
                item["verdicts"] = {c: data.get("verdicts", {}).get(c, "unsure")
                                    for c in claims}
            except Exception:  # noqa: BLE001: verdicts are best-effort
                item["verdicts"] = {c: "unsure" for c in claims}
        return item

    from concurrent.futures import ThreadPoolExecutor

    # The proof is for the photos on the screen, in the order on the screen.
    shown = rec.get("final_ids") or []
    by_id = {c.image_id: c for c in s.candidates}
    top = [by_id[i] for i in shown if i in by_id][:3] or s.candidates[:3]
    with ThreadPoolExecutor(max_workers=3) as ex:
        out = list(ex.map(check, top))
    if out and all(item.get("error") for item in out):
        raise HTTPException(503, "Evidence is not available right now.")
    return {"session_id": session_id, "evidence": out,
            "claims_checked": {"colour": bool(colour),
                               "count": said_count is not None,
                               "extra": claims}}


@app.get("/api/random")
def random_images(corpus: str = "coco_val2017", n: int = 12, seed: int = 7) -> dict:
    """Target-picker support: a deterministic pseudo-random grid.
    The pick itself stays in the browser (localStorage); the
    server never learns which image was chosen."""
    import random as _random

    index = get_index(corpus)
    entries = list(index._entries)  # type: ignore[attr-defined]
    picks = _random.Random(seed).sample(entries, min(n, len(entries)))
    return {"corpus": corpus, "items": [
        {"image_id": e.image_id, "file": Path(e.path).name,
         "url": f"/api/image/{corpus}/{e.image_id}"} for e in picks]}


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
