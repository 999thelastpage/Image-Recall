"""Versioned configuration defaults. Bump CONFIG_VERSION on any change
that alters ranking behaviour so cached results can be invalidated."""

CONFIG_VERSION = 1

# Retrieval
BRANCH_TOP_K = 50  # reserved per-branch candidate quota
FINAL_TOP_K = 5
EMBEDDING_DIM = 512  # facebook/metaclip-b32-fullcc2.5b output size

# Sessions
MAX_QUESTIONS = 3

# Encoder (Meta open weights; frozen at G0 — see _context/DECISIONS.md)
ENCODER_MODEL = "facebook/metaclip-b32-fullcc2.5b"
