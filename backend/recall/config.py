"""Versioned configuration defaults. Bump CONFIG_VERSION on any change
that alters ranking behaviour so cached results can be invalidated."""

CONFIG_VERSION = 1

# Retrieval
BRANCH_TOP_K = 50  # reserved per-branch candidate quota
FINAL_TOP_K = 5
EMBEDDING_DIM = 768  # facebook/dinov2-base output size

# Sessions
MAX_QUESTIONS = 3

# Encoder (Meta open weights; licence verified at ingest time)
ENCODER_MODEL = "facebook/dinov2-base"
