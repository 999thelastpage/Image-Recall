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
- **MetaCLIP-L14** (`facebook/metaclip-l14-fullcc2.5b`, Meta open
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

Gate G2 (30 cases: 10 wrong-colour, 10 wrong-object, 10
accurate; simulated truthful user, ≤3 questions): wrong-clue
Hit@5 **10/20 → 13/20** after the session loop. Colour repair
5→8. A first run failed (9/20) because dropping the main object
noun destroys a query; the shipped rule is now *modifiers are
dropped, anchors are replaced*, with a regression test.

Corpus scale vs quality (100 own-caption queries per corpus,
`scripts/quality_check.py`). Encoder upgraded B32 → L14 on
2026-10-30 after benchmarking B32 / L14 / H14 on both scales:

| Corpus | Encoder | R@1 | R@5 | R@10 | R@50 | Median rank |
|---|---|---|---|---|---|---|
| 5,000 | B32 | 0.30 | 0.54 | 0.67 | 0.89 | 4 |
| 5,000 | L14 | 0.39 | 0.64 | 0.72 | 0.92 | 2 |
| 82,783 | B32 | 0.07 | 0.17 | 0.26 | 0.48 | 56 |
| 82,783 | **L14 (live)** | **0.17** | **0.32** | 0.36 | 0.53 | 48 |

L14 nearly doubles top-5 accuracy at 83k — and 0.32 is still
weak in absolute terms. That remaining gap is what the repair
loop, Muse reranking, and SAM evidence are for. Known failure
layers are tracked per result: parsing / candidate coverage /
evidence / planning / ranking. Note: MetaCLIP weights are
CC-BY-NC (hackathon-fine); the Apache-2.0 PE-Core encoder is the
commercial-path candidate.

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

## Status & next

Done: G0 (API + encoder gates), Day 1 engine (Gate G1), encoder
upgrade B32→L14, Day 2 session loop + target picker (Gate G2).
Live corpora in this workspace: 5,000 COCO val + 82,783 COCO
train2014, both on MetaCLIP-L14. Tests: 20 passing.

Next (Day 3): Muse rerank + SAM evidence inside the loop,
hosted demo (Modal/Baseten), 2.5-minute video, submission.
Parked by design: sub-minute ingest (reduced-resolution dataset
+ fused parallel pipeline) — planned, measured, awaiting a
go/no-go discussion. Known weaknesses, honestly: wrong-object
repair is at parity (coverage-bound), union recall at 83k is
unmeasured, MetaCLIP weights are CC-BY-NC (PE-Core is the
Apache-2.0 commercial candidate).
