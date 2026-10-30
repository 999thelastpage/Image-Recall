"""Ingestion manifest: one JSON object per image with checksum and
row mapping. The manifest is the source of truth linking an
embedding row back to an image file."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ManifestEntry:
    image_id: str
    path: str
    sha256: str
    size_bytes: int
    row: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(image_dir: Path) -> list[ManifestEntry]:
    """Scan a directory deterministically (sorted paths)."""
    files = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    entries: list[ManifestEntry] = []
    for row, path in enumerate(files):
        entries.append(
            ManifestEntry(
                image_id=f"img-{row:06d}",
                path=str(path),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
                row=row,
            )
        )
    return entries


def write_manifest(entries: list[ManifestEntry], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")


def read_manifest(path: Path) -> list[ManifestEntry]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(ManifestEntry(**json.loads(line)))
    return entries
