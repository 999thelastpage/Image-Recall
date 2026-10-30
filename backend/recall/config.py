"""Versioned configuration defaults. Bump CONFIG_VERSION on any change
that alters ranking behaviour so cached results can be invalidated."""

CONFIG_VERSION = 1

# Retrieval
BRANCH_TOP_K = 50  # reserved per-branch candidate quota (small corpora)


def branch_quota(corpus_size: int) -> int:
    """Per-branch quota scales with corpus size: a fixed top-50
    starves the union at 83k+ (measured failure layer: coverage)."""
    if corpus_size <= 10_000:
        return 50
    if corpus_size <= 100_000:
        return 150
    return 250
FINAL_TOP_K = 5
EMBEDDING_DIM = 768  # facebook/metaclip-l14-fullcc2.5b output size

# Sessions
MAX_QUESTIONS = 3

# Encoder (Meta open weights; B32 frozen at G0, upgraded to L14
# 2026-10-30: 83k R@5 0.17 -> 0.32 — see _context/DECISIONS.md)
ENCODER_MODEL = "facebook/metaclip-l14-fullcc2.5b"
