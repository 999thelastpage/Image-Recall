"""Modal deployment for the Recall console (Day-3 hosting).

STATUS: deploy-ready, NOT yet deployed — needs a Modal account
token (modal token new). One-time data prep before first deploy:

    python -m scripts.ingest --images data/coco/val2017 --name coco_val2017 --thumbs
    # serves thumbnails (data/thumbs/) so the hosted demo carries
    # ~30 MB of images instead of 788 MB of originals.

Deploy:
    modal deploy deploy/modal_app.py
Then put the printed URL in README, the video description, and
the submission Project Description. Keep it alive Nov 5-9.
"""

from __future__ import annotations

import modal

app = modal.App("recall-console")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "uvicorn[standard]", "numpy", "Pillow",
                 "torch", "transformers", "python-multipart")
    .add_local_dir("backend", "/srv/backend")
    .add_local_dir("frontend", "/srv/frontend")
    .add_local_dir("embeddings", "/srv/embeddings")
    .add_local_dir("data/manifests", "/srv/data/manifests")
    .add_local_dir("data/thumbs", "/srv/data/thumbs")
)


@app.function(image=image, gpu="T4", timeout=900,
              secrets=[modal.Secret.from_name("recall-secrets")])
@modal.asgi_app()
def web():
    import os
    import sys

    sys.path.insert(0, "/srv")
    os.chdir("/srv")
    from backend.app import app as fastapi_app

    return fastapi_app
