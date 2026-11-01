"""Gate G2: 30-case smoke eval vs plain baseline.

10 wrong-colour + 10 wrong-object + 10 accurate cases (COCO val
captions). A simulated truthful user answers: when asked which
detail they might be misremembering, they name the corrupted one
(a cooperative user who is genuinely unsure of it); attribute
questions about it get the TRUE value; everything else "Not sure".
The target id is used only for scoring, never by the session.

Gate: on the 20 wrong-clue cases, Recall final Hit@5 > baseline.

    .venv/bin/python -m scripts.g2_eval
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLOUR_WRONG = {"red": "blue", "orange": "purple", "yellow": "green",
                "green": "red", "blue": "orange", "purple": "yellow",
                "pink": "brown", "brown": "black", "black": "white",
                "white": "black", "gray": "pink"}
OBJECT_WRONG = {"dog": "cat", "cat": "dog", "car": "bus", "bus": "car",
                "horse": "cow", "bicycle": "motorcycle", "boat": "train",
                "train": "boat", "bird": "plane", "chair": "table"}


def corrupt(cap: str, mapping: dict) -> tuple[str, str, str] | None:
    for true_w, wrong_w in mapping.items():
        if re.search(rf"\b{true_w}\b", cap, re.I):
            out = re.sub(rf"\b{true_w}\b",
                         lambda m: wrong_w.capitalize() if m.group(0)[0].isupper() else wrong_w,
                         cap, count=1, flags=re.I)
            return out, true_w, wrong_w
    return None


def pick_cases(captions: dict, by_file: dict, per_kind: int = 10,
               n_accurate: int = 10) -> list[dict]:
    colour, obj, acc = [], [], []
    for fname in sorted(captions):
        if fname not in by_file:
            continue
        cap = captions[fname]
        c = corrupt(cap, COLOUR_WRONG)
        if c and len(colour) < 90:
            colour.append({"file": fname, "kind": "colour", "query": c[0],
                           "true": c[1], "wrong": c[2]})
        o = corrupt(cap, OBJECT_WRONG)
        if o and len(obj) < 90:
            obj.append({"file": fname, "kind": "object", "query": o[0],
                        "true": o[1], "wrong": o[2]})
        if len(acc) < 90 and len(cap.split()) >= 6:
            acc.append({"file": fname, "kind": "accurate", "query": cap,
                        "true": "", "wrong": ""})
    spread = lambda xs, n: xs[:: max(1, len(xs) // n)][:n]  # noqa: E731
    return spread(colour, per_kind) + spread(obj, per_kind) + spread(acc, n_accurate)


def classify_failure(base_rank, union_rank, final_rank, hit_k: int = 5) -> str:
    """Failure layer for one case. Ceiling vs coverage vs ranking
    keeps the next lever honest instead of tuning blind."""
    if final_rank is not None and final_rank <= hit_k:
        return "hit"
    if base_rank is not None and base_rank <= hit_k:
        return "ranking-demotion"
    if union_rank is None:
        return "coverage"
    if union_rank > 40:
        return "pool-depth"
    return "ranking"


def _rank(session, target: str):
    return next((r for r, c in enumerate(session.candidates, 1)
                 if c.image_id == target), None)


def simulate(session, case: dict, target: str | None = None,
             stages: list | None = None) -> None:
    from backend.recall.questions import NOT_SURE, SOMETHING_ELSE, YES_SURE

    while session.pending is not None and len(session.answers) < 3:
        q = session.pending
        value = NOT_SURE
        if q.kind == "subject":
            if (case["wrong"] and q.clue_index is not None
                    and 0 <= q.clue_index < len(session.clues)
                    and case["wrong"] in session.clues[q.clue_index].text.lower()):
                value = SOMETHING_ELSE
            else:
                value = YES_SURE
        elif q.kind in ("category", "details"):
            value = NOT_SURE
        elif q.kind == "dimension":
            # A truthful user blames the colour only when it is wrong:
            # with a wrong noun in 'purple car', the colour is fine.
            if case["kind"] == "colour" and q.clue_index is not None \
                    and 0 <= q.clue_index < len(session.clues) \
                    and case["wrong"] in session.clues[q.clue_index].text.lower():
                value = SOMETHING_ELSE
            else:
                value = YES_SURE
        elif q.kind == "uncertainty" and case["wrong"]:
            value = next((o for o in q.options
                          if case["wrong"] in o.lower()), NOT_SURE)
        elif q.kind == "attribute" and q.clue_index is not None and case["wrong"]:
            clue = session.clues[q.clue_index]
            if case["wrong"] in clue.text.lower() and case["true"] \
                    and case["true"] in q.options:
                value = case["true"]
        session.answer(value)
        if stages is not None and target is not None:
            stages.append(_rank(session, target))


def main() -> int:
    import argparse

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=10,
                    help="wrong-colour and wrong-object cases each")
    ap.add_argument("--offline-parse", action="store_true",
                    help="use the lexicon parser even when a Muse key is set")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N cases (cost check)")
    ap.add_argument("--rerank", action="store_true",
                    help="apply the product's doubt-aware top-40 rerank")
    args = ap.parse_args()

    from backend.recall.clues import Clue
    from backend.recall.embeddings import MetaClipEncoder
    from backend.recall.manifest import read_manifest
    from backend.recall.providers import (
        MuseClient,
        ProviderDisabled,
        load_dotenv,
        rerank_client,
    )
    from backend.recall.retrieve import ExactIndex
    from backend.recall.session import SearchSession

    load_dotenv()
    entries = read_manifest(Path("data/manifests/coco_val2017.jsonl"))
    index = ExactIndex([e.image_id for e in entries],
                       np.load("embeddings/coco_val2017.npy"))
    caps = json.loads(Path("data/coco/captions.json").read_text())
    by_file = {Path(e.path).name: e.image_id for e in entries}
    cases = pick_cases(caps, by_file, per_kind=args.per_kind)
    if args.limit:
        cases = cases[:: max(1, len(cases) // args.limit)][:args.limit]
    print(f"{len(cases)} cases "
          f"({sum(c['kind'] == 'colour' for c in cases)} colour, "
          f"{sum(c['kind'] == 'object' for c in cases)} object, "
          f"{sum(c['kind'] == 'accurate' for c in cases)} accurate)")

    enc = MetaClipEncoder()
    try:
        muse = MuseClient()
    except ProviderDisabled:
        muse = None
    if args.offline_parse:
        muse = None
    reranker = None
    if args.rerank:
        reranker = rerank_client()
    print("parse:", muse.model if muse else "OFFLINE fallback (no Muse calls)")
    print("rerank:", reranker.model if reranker else "off", flush=True)
    calls = failed = 0
    rows = []
    for i, case in enumerate(cases, 1):
        target = by_file[case["file"]]
        base_rank = index.rank_of(enc.encode_texts([case["query"]]), target)
        base_hit = base_rank is not None and base_rank <= 5
        if muse is None:
            from backend.recall.clues import dedupe_clues, parse_fallback

            clues = dedupe_clues(parse_fallback(case["query"]))
        else:
            chips = muse.parse(case["query"])
            clues = [Clue(c["text"], c.get("kind", "other"), c.get("referent", ""))
                     for c in chips if len(c["text"].strip()) > 1]
        s = SearchSession(index, enc, case["query"], clues)
        s.start()
        union_rank = _rank(s, target)
        stages: list = []
        simulate(s, case, target=target, stages=stages)
        pre_rerank_rank = _rank(s, target)
        out = s.finish()
        final_rank = next((r for r, x in enumerate(out["results"], 1)
                           if x["image_id"] == target), None)
        # final rank within full candidate list, not just top-5 print
        full_rank = next((r for r, c in enumerate(s.candidates, 1)
                          if c.image_id == target), None)
        if reranker is not None:
            from backend.recall.questions import NOT_SURE
            from backend.recall.rerank import MuseReranker, doubt_aware_description

            dropped = [s.clues[i].text for i in sorted(s.dropped)]
            corrections = [
                (s.clues[a.clue_index].text, a.value) for a in s.answers
                if a.kind == "attribute" and a.clue_index is not None
                and a.value != NOT_SURE]
            desc = doubt_aware_description(case["query"], dropped, corrections)
            paths = {e.image_id: e.path for e in entries}
            from backend.recall.rerank import RERANK_TOP_N, RERANK_WEIGHT

            ranked = MuseReranker(reranker, max_workers=8).rerank(
                s.candidates, paths, desc,
                top_n=RERANK_TOP_N, weight=RERANK_WEIGHT)
            head = ranked[:RERANK_TOP_N]
            calls += len(head)
            failed += sum(x.muse is None for x in ranked if x.final != -1.0)
            full_rank = next((r for r, x in enumerate(ranked, 1)
                              if x.image_id == target), None)
        recall_hit = full_rank is not None and full_rank <= 5
        layer = classify_failure(base_rank, union_rank, full_rank)
        rows.append({**case, "base_hit": base_hit, "recall_hit": recall_hit,
                     "base_rank": base_rank, "final_rank": full_rank,
                     "union_rank": union_rank, "layer": layer,
                     "stages": stages, "pre_rerank_rank": pre_rerank_rank})
        print(f"[{i:2d}] {case['kind']:8s} base={str(base_rank):>5s} "
              f"union={str(union_rank):>5s} stages={stages} "
              f"pre={str(pre_rerank_rank):>5s} final={str(full_rank):>5s} "
              f"layer={layer} {'REPAIRED' if recall_hit and not base_hit else ''} "
              f"| {case['query'][:58]}")

    for kind, label in [("colour", "wrong-colour"), ("object", "wrong-object"),
                        ("accurate", "accurate")]:
        sub = [r for r in rows if r["kind"] == kind]
        b = sum(r["base_hit"] for r in sub)
        r_ = sum(r["recall_hit"] for r in sub)
        b10 = sum((r["base_rank"] or 9999) <= 10 for r in sub)
        r10 = sum((r["final_rank"] or 9999) <= 10 for r in sub)
        print(f"{label}: baseline Hit@5 {b}/{len(sub)} -> Recall {r_}/{len(sub)} "
              f"| Hit@10 {b10} -> {r10}")
    from collections import Counter

    print("failure layers:", dict(Counter(r["layer"] for r in rows)))
    if reranker is not None:
        print(f"rerank calls {calls}, failed scores {failed}")
    for r in rows:
        if r["layer"] == "ranking-demotion":
            src = "rerank" if (r["pre_rerank_rank"] or 999) <= 5 else "answers"
            print(f"demotion via {src}: union={r['union_rank']} "
                  f"stages={r['stages']} pre={r['pre_rerank_rank']} "
                  f"final={r['final_rank']} | {r['file']}")
    wrong = [r for r in rows if r["kind"] != "accurate"]
    b = sum(r["base_hit"] for r in wrong)
    r_ = sum(r["recall_hit"] for r in wrong)
    print(f"\nGATE G2 (wrong-clue): baseline {b}/{len(wrong)} -> Recall {r_}/{len(wrong)}")
    print("G2 PASS" if r_ > b else "G2 FAIL")
    return 0 if r_ > b else 1


if __name__ == "__main__":
    raise SystemExit(main())
