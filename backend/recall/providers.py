"""Hosted Meta Model API providers (Muse Spark, SAM 3.1).

Both are DISABLED unless credentials exist in the environment
(META_MODEL_API_KEY, normally loaded from the local .env — never
from chat, never committed). The retrieval core runs fully locally
without them. Endpoints verified at G0 (2026-10-30):
  Muse Spark 1.3: POST https://api.meta.ai/v1/chat/completions
  SAM 3.1:        POST https://api.meta.ai/v1/responses
Muse is a reasoning model: keep max_tokens generous (>=2000) and
reasoning_effort "low" for structured tasks, else content is null.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.meta.ai/v1"
MUSE_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"

_PARSE_SYSTEM = (
    "Parse a photo description into clue chips. Return ONLY JSON: "
    '{"clues":[{"text":...,"kind":"colour|object|count|setting|'
    'activity|person|other","referent":...}]}. Each clue is a '
    "short meaningful phrase (e.g. 'red jacket', 'two dogs', "
    "'at the beach') — never filler words (a, the, some) and never "
    "a bare verb. A colour together with its object is ONE clue "
    "('purple bus'). Bind counts to their noun. Return 3 to 6 "
    "clues covering the distinct facts (subject, appearance, "
    "setting, activity) so each can be doubted separately. "
    "No prose."
)


def _api_key() -> str | None:
    return os.environ.get("META_MODEL_API_KEY") or None


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (names/values stay local, never logged)."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()  # noqa: SIM115
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _post(path: str, payload: dict, timeout: int = 120,
          base_url: str = BASE_URL, api_key: str | None = None) -> dict:
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key or _api_key()}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if attempt == 2 or (e.code != 429 and e.code < 500):
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


class ProviderDisabled(RuntimeError):
    pass


class MuseClient:
    """Muse Spark: parse / plan / explain. Disabled without a key."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None) -> None:
        # MUSE_BASE_URL and MUSE_API_KEY reach Muse through another host.
        self.base_url = (base_url or os.environ.get("MUSE_BASE_URL")
                         or BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("MUSE_API_KEY") or None
        if not (self.api_key or _api_key()):
            raise ProviderDisabled(
                "META_MODEL_API_KEY not set; Muse provider disabled. "
                "Add it to your local .env (see .env.example)."
            )
        # Read here, not at import: load_dotenv runs after this module loads.
        self.model = model or os.environ.get("MUSE_MODEL") or MUSE_MODEL

    def _reasoning(self) -> dict:
        """OpenRouter takes the effort in a nested field."""
        if "openrouter.ai" in self.base_url:
            return {"reasoning": {"effort": "low"}}
        return {"reasoning_effort": "low"}

    def _chat(self, messages: list[dict], max_tokens: int = 2000) -> str:
        out = _post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                **self._reasoning(),
                "temperature": 0,  # deterministic: eval scores must not drift run to run
            },
            base_url=self.base_url, api_key=self.api_key,
        )
        return out["choices"][0]["message"]["content"] or ""

    def chat_with_image(self, image_path: str, prompt: str, max_tokens: int = 800) -> str:
        raw = open(image_path, "rb").read()  # noqa: SIM115
        b64 = base64.b64encode(raw).decode()
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        out = _post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                **self._reasoning(),
                "temperature": 0,
            },
            base_url=self.base_url, api_key=self.api_key,
        )
        return out["choices"][0]["message"]["content"] or ""

    def parse(self, description: str) -> list[dict]:
        """Description -> validated clue chips [{text,kind,referent}]."""
        data: dict | None = None
        for attempt in range(2):  # reasoning models occasionally return empty or broken JSON
            txt = self._chat(
                [
                    {"role": "system", "content": _PARSE_SYSTEM},
                    {"role": "user", "content": description},
                ],
                max_tokens=2000 + 1000 * attempt,
            )
            try:
                data = json.loads(txt[txt.index("{") : txt.rindex("}") + 1])
                break
            except ValueError:
                continue
        if data is None:
            raise ValueError(f"Muse returned no usable JSON for: {description!r}")
        clues = []
        for c in data.get("clues", []):
            if isinstance(c, dict) and str(c.get("text", "")).strip():
                clues.append(
                    {
                        "text": str(c["text"]).strip(),
                        "kind": str(c.get("kind", "other")),
                        "referent": str(c.get("referent", "")),
                    }
                )
        if not clues:
            raise ValueError(f"Muse returned no clues for: {description!r}")
        # dedupe: drop exact repeats and chips contained in a longer chip
        out: list[dict] = []
        for c in clues:
            t = c["text"].strip().lower()
            if any(o["text"].strip().lower() == t for o in out):
                continue
            if any(t in o["text"].strip().lower() and len(o["text"]) > len(c["text"])
                   and (o.get("kind") == c.get("kind")
                        or "other" in (o.get("kind"), c.get("kind")))
                   for o in clues if o is not c):
                continue
            out.append(c)
        return out


def rerank_client() -> MuseClient:
    """Client for reranking only. RERANK_BASE_URL, RERANK_API_KEY and
    RERANK_MODEL point it at a stand-in model for local testing; without
    them it is Muse. Parsing, verdicts and SAM never use these."""
    base_url = os.environ.get("RERANK_BASE_URL")
    if not base_url:
        return MuseClient()
    return MuseClient(os.environ.get("RERANK_MODEL") or None,
                      base_url=base_url.rstrip("/"),
                      api_key=os.environ.get("RERANK_API_KEY"))


class BackupReranker:
    """Backup for the AI check: NVIDIA Nemotron 3.5 Rerank. One request scores
    many photos. Disabled without BACKUP_RERANK_API_KEY."""

    BASE = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b"
    model = "nvidia/nemotron-3.5-rerank-8b"
    CHUNK = 10

    def __init__(self) -> None:
        self.api_key = os.environ.get("BACKUP_RERANK_API_KEY") or None
        if not self.api_key:
            raise ProviderDisabled("BACKUP_RERANK_API_KEY not set")

    def _score_chunk(self, description: str, paths: list[str]) -> dict[str, float]:
        passages = []
        for p in paths:
            mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(open(p, "rb").read()).decode()  # noqa: SIM115
            passages.append({"image": f"data:{mime};base64,{b64}"})
        out = _post("/reranking", {"model": self.model, "query": {"text": description},
                                   "passages": passages},
                    base_url=self.BASE, api_key=self.api_key)
        return {paths[r["index"]]: float(r["logit"]) for r in out["rankings"]}

    def logits(self, description: str, paths: list[str]) -> dict[str, float]:
        """Raw relevance per photo. A failed chunk is left out."""
        from concurrent.futures import ThreadPoolExecutor

        def safe(chunk: list[str]) -> dict[str, float]:
            try:
                return self._score_chunk(description, chunk)
            except Exception:  # noqa: BLE001: those photos stay neutral
                return {}

        chunks = [paths[i : i + self.CHUNK] for i in range(0, len(paths), self.CHUNK)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(safe, chunks))
        return {k: v for r in results for k, v in r.items()}


class SamClient:
    """SAM 3.1: grounded segmentation evidence. Disabled without a key."""

    def __init__(self, model: str = SAM_MODEL) -> None:
        if not _api_key():
            raise ProviderDisabled(
                "META_MODEL_API_KEY not set; SAM provider disabled. "
                "Add it to your local .env (see .env.example)."
            )
        self.model = model

    def segment(self, image_path: str, prompt: str) -> dict:
        """Segment `prompt` in an image. Returns the raw output text
        (box + encoded mask in SAM's payload format) plus latency."""
        raw = open(image_path, "rb").read()  # noqa: SIM115
        b64 = base64.b64encode(raw).decode()
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        t0 = time.time()
        out = _post(
            "/responses",
            {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
            },
        )
        text = ""
        for item in out.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
        return {"text": text, "seconds": round(time.time() - t0, 2), "raw": out}


def providers_enabled() -> bool:
    return bool(_api_key() or os.environ.get("MUSE_API_KEY"))
