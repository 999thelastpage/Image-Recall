"""Modal deployment for the Recall console.

STATUS: deploy-ready, NOT yet deployed — needs the builder's Modal
account. One-time data prep before first deploy (already run
locally, artefacts are git-ignored and uploaded by the image build):

    python -m scripts.prep_demo   # 5,000 thumbs + demo manifest

The hosted demo serves ONLY coco_val2017 (5,000 photos, thumbnails
at <=384px, ~117 MB). It cannot serve train2014: 254 MB of
embeddings plus photos that only exist on the builder's disk.

Deploy (builder's terminal, once, after `modal token new` and after
creating the `recall-secrets` secret with META_MODEL_API_KEY):

    pip install modal
    modal deploy deploy/modal_app.py

Then put the printed URL in README, the video description, and the
submission Project Description. Keep it alive Nov 5-9.
"""

from __future__ import annotations

import modal

app = modal.App("recall-console")

ENCODER_MODEL = "facebook/metaclip-l14-fullcc2.5b"  # backend.recall.config


def _download_encoder() -> None:
    """Bake the encoder weights into the image so a cold container
    never downloads 1.7 GB while a judge waits."""
    from transformers import AutoProcessor, CLIPModel

    CLIPModel.from_pretrained(ENCODER_MODEL)
    AutoProcessor.from_pretrained(ENCODER_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "uvicorn[standard]", "numpy", "Pillow",
                 "torch", "transformers", "python-multipart")
    .env({"HF_HOME": "/opt/hf"})
    .run_function(_download_encoder)
    .add_local_dir("backend", "/srv/backend")
    .add_local_dir("frontend", "/srv/frontend")
    # Only the demo corpus ships — not the 706 MB embeddings folder.
    .add_local_file("embeddings/coco_val2017.npy",
                    "/srv/embeddings/coco_val2017.npy")
    .add_local_file("embeddings/coco_val2017.meta.json",
                    "/srv/embeddings/coco_val2017.meta.json")
    .add_local_file("data/manifests/coco_val2017.demo.jsonl",
                    "/srv/data/manifests/coco_val2017.jsonl")
    .add_local_dir("data/thumbs/coco_val2017",
                   "/srv/data/thumbs/coco_val2017")
)


@app.function(image=image, gpu="T4", memory=4096, timeout=900,
              max_containers=1, scaledown_window=600,
              env={"RECALL_DEMO": "1"},
              secrets=[modal.Secret.from_name("recall-secrets")])
@modal.asgi_app()
def web():
    import os
    import sys

    sys.path.insert(0, "/srv")
    os.chdir("/srv")
    from backend.app import app as fastapi_app

    return fastapi_app
