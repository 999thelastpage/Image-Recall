# Recall — finds the photo you remember wrong

Describe a photo with one detail mistaken — a colour, a count, a
place — and Recall doubts each detail in turn, searches alternate
interpretations, grounds what it sees, and asks up to three
one-tap questions to land your photo in its top five.

**Live demo:** _to be added once hosted (Modal/Baseten)._

## How it works

- **Muse Spark 1.3** (Meta Model API) parses a free description
  into clue chips, picks each clarifying question, and writes
  grounded explanations.
- **SAM 3.1** (Meta Model API) segments the referred object in top
  candidates; colour/count evidence is sampled inside the mask and
  shown to the user ("you said red — this jacket is orange").
- **Meta open-weights vision encoder** provides image embeddings
  for retrieval (model + revision pinned in the ingest manifest).
- Our orchestration: leave-one-clue-out hypotheses, per-branch
  candidate union with reserved quotas, soft scoring (no hard
  filters; unknown evidence is neutral), whole-corpus re-entry
  after each answer, ≤3-question session cap.

## Status

Day 1: ingestion + baseline retrieval vertical slice. Measured
results will replace this line as gates pass; nothing here is a
fabricated benchmark.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env   # fill in your own keys; never commit .env
python -m scripts.ingest --help
python -m scripts.demo_search "a red jacket on a bicycle"
```

Hosted providers are disabled unless credentials are present in
`.env`; the retrieval core runs fully locally.
