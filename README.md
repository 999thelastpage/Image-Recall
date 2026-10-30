# Recall — finds the photo you remember wrong

Describe a photo with one detail mistaken — a colour, a count, a
place — and Recall doubts each detail in turn, searches alternate
interpretations, grounds what it sees, and asks up to three
one-tap questions to land your photo in its top five.

Built for the Meta Global AI Developer Hackathon (Indie track,
Build for Your World). Solo project.

**Live demo:** _hosted link to be added (Modal/Baseten)._ Until
then it runs locally in one command (below) — that local console
is what the demo video will show.

## Try it now (local console)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env   # optional: add META_MODEL_API_KEY for Muse parsing
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — pick a corpus (two are pre-ingested
in this workspace: 5k COCO val, 82,783 COCO train2014), or add
your own photos:

- **Choose folder…** — system picker. Photos are sent to the local
  server (127.0.0.1 only), embedded, and afterwards only small
  thumbnails + embeddings (~2 KB/photo) are kept by default, so a
  5 GB folder does not become another 5 GB. A checkbox keeps
  full-size copies if you want them.
- **Paste a path** — Windows (`E:\photos\trip`) or Linux
  (`/home/you/photos`) paths both work; reads photos in place,
  zero copying. Best for large folders.

Then ask in plain language — “a man riding a bicycle next to a red
car”, “the cake on the kitchen table” — and compare **plain
search** against **Recall repair** side by side, with per-query
timings, clue chips, and rescued candidates outlined in green.

## How it works

- **Muse Spark 1.3** (Meta Model API) parses a free description
  into clue chips, picks each clarifying question, and writes
  grounded explanations. Disabled cleanly when no key is set.
- **SAM 3.1** (Meta Model API) segments the referred object in top
  candidates; colour/count evidence is sampled inside the mask and
  shown (“you said red — this jacket is orange”).
- **MetaCLIP-B32** (`facebook/metaclip-b32-fullcc2.5b`, Meta open
  weights) embeds images and text for exact cosine retrieval.
- Our orchestration: leave-one-clue-out hypotheses, per-branch
  candidate union with reserved quotas, soft scoring (no hard
  filters; unknown evidence is neutral), ≤3-question sessions.

## Measured results (no fabricated numbers)

G0 smoke (500 COCO images): Recall@50 0.998, 164 img/s on an
RTX 4070. Muse parse: 10/10 messy descriptions → usable clue
chips. SAM 3.1: exact box in 1.8s.

Gate G1 (10 hand-made wrong-colour cases, 5k corpus): branch
union contains the target **9/10** vs plain-search Hit@5 6/10.

Corpus scale vs quality (100 own-caption queries per corpus,
`scripts/quality_check.py`):

| Corpus | R@1 | R@5 | R@10 | R@50 | Median rank |
|---|---|---|---|---|---|
| 5,000 | 0.30 | 0.54 | 0.67 | 0.89 | 4 |
| 82,783 | 0.07 | 0.17 | 0.26 | 0.48 | 56 |

Plain search weakens sharply at 83k — that gap is what the repair
loop and a stronger encoder (PE-Core-L14 / MetaCLIP-H14, same
harness) are for. Known failure layers are tracked per result:
parsing / candidate coverage / evidence / planning / ranking.

## Layout

```
backend/    FastAPI app + recall package (manifest, embeddings,
            retrieval, clues, ranking, Meta providers)
frontend/   static test console (no build step)
scripts/    ingest, demo_search, g0_encoder, g1_eval,
            quality_check, fetch_coco, make_fixture
tests/      index correctness, hypotheses, union, UI helpers
data/       manifests tracked; images/embeddings git-ignored
```

## Honesty rules

No fabricated benchmarks, no target-ID leakage (targets are used
only by offline scoring scripts), hosted providers off without
credentials, generated images always labelled. Failures are
reported with their layer, not hidden.

## Next

Day 2: question engine + SAM evidence in the loop, Race view
(rank climbing live), 30-case Gate G2. Day 3: hosted demo, video,
submission. Ingest-speed work (reduced-resolution dataset +
parallel pipeline) is designed and parked pending evaluation.
