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


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class ProviderDisabled(RuntimeError):
    pass


class MuseClient:
    """Muse Spark: parse / plan / explain. Disabled without a key."""

    def __init__(self, model: str = MUSE_MODEL) -> None:
        if not _api_key():
            raise ProviderDisabled(
                "META_MODEL_API_KEY not set; Muse provider disabled. "
                "Add it to your local .env (see .env.example)."
            )
        self.model = model

    def _chat(self, messages: list[dict], max_tokens: int = 2000) -> str:
        out = _post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "reasoning_effort": "low",
                "temperature": 0,  # deterministic: eval scores must not drift run to run
            },
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
                "reasoning_effort": "low",
                "temperature": 0,
            },
        )
        return out["choices"][0]["message"]["content"] or ""

    def parse(self, description: str) -> list[dict]:
        """Description -> validated clue chips [{text,kind,referent}]."""
        txt = ""
        for attempt in range(2):  # reasoning models occasionally return empty
            txt = self._chat(
                [
                    {"role": "system", "content": _PARSE_SYSTEM},
                    {"role": "user", "content": description},
                ],
                max_tokens=2000 + 1000 * attempt,
            )
            if "{" in txt:
                break
        data = json.loads(txt[txt.index("{") : txt.rindex("}") + 1])
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
    return _api_key() is not None
