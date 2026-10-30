"""Hosted Meta Model API providers (Muse Spark, SAM 3.1).

Both are DISABLED unless credentials exist in the environment
(META_MODEL_API_KEY, read from .env by the caller — never from
chat, never committed). The retrieval core runs fully locally
without them.
"""

from __future__ import annotations

import os


def _api_key() -> str | None:
    return os.environ.get("META_MODEL_API_KEY") or None


class ProviderDisabled(RuntimeError):
    pass


class MuseClient:
    """Muse Spark: parse / plan / explain. Disabled without a key."""

    def __init__(self) -> None:
        if not _api_key():
            raise ProviderDisabled(
                "META_MODEL_API_KEY not set; Muse provider disabled. "
                "Add it to your local .env (see .env.example)."
            )
        raise NotImplementedError("wire Meta Model API call at G0")


class SamClient:
    """SAM 3.1: grounded segmentation evidence. Disabled without a key."""

    def __init__(self) -> None:
        if not _api_key():
            raise ProviderDisabled(
                "META_MODEL_API_KEY not set; SAM provider disabled. "
                "Add it to your local .env (see .env.example)."
            )
        raise NotImplementedError("wire Meta Model API call at G0")


def providers_enabled() -> bool:
    return _api_key() is not None
