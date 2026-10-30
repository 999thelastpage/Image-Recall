"""Recall search session: the ≤3-question repair loop.

State machine: new -> asking (<= MAX_QUESTIONS) -> done.
Answers never hard-filter. Dropping a clue raises the weight of
its leave-one-out branch and triggers whole-corpus re-entry with
the revised query; "Not sure" is neutral. Explanations are
templated from stored provenance (branches + answers), never
invented.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from .clues import Clue, Hypothesis, hypotheses
from .config import BRANCH_TOP_K, MAX_QUESTIONS
from .questions import NOT_SURE, Question, attribute_template, uncertainty_question
from .rank import Candidate, branch_union, weighted_rerank
from .retrieve import ExactIndex

W_DROPPED = 1.6   # branch weight when user doubts that clue
W_CONFIRMED = 0.5  # branch weight for dropping a confirmed clue
ANCHOR_KINDS = {"object", "person"}  # doubted anchors get replaced, not dropped


@dataclass
class Answer:
    qid: str
    kind: str
    value: str
    clue_index: int | None


@dataclass
class SearchSession:
    index: ExactIndex
    encoder: object  # has encode_texts(list[str]) -> np.ndarray
    description: str
    clues: list[Clue]
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    candidates: list[Candidate] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    answers: list[Answer] = field(default_factory=list)
    pending: Question | None = None
    dropped: set[int] = field(default_factory=set)
    doubted_anchor: int | None = None
    status: str = "new"  # new | asking | done

    # -- setup ----------------------------------------------------
    def start(self) -> None:
        hyps = hypotheses(self.clues)
        queries = [(h, self.encoder.encode_texts([h.query_text])) for h in hyps]
        self.candidates = branch_union(self.index, queries, BRANCH_TOP_K)
        self.status = "asking"
        self.pending = self._plan()

    # -- planning ---------------------------------------------------
    def _plan(self) -> Question | None:
        n = len(self.answers) + 1
        if n > MAX_QUESTIONS:
            return None
        asked_kinds = [a.kind for a in self.answers]
        # a doubted anchor must be replaced before anything else
        if self.doubted_anchor is not None and all(
            a.clue_index != self.doubted_anchor for a in self.answers
        ):
            return attribute_template(self.clues[self.doubted_anchor],
                                      self.doubted_anchor, n)
        # 1st: uncertainty, unless already dropped the max clues
        if "uncertainty" not in asked_kinds and len(self.clues) >= 2:
            return uncertainty_question(self.clues, -1, n)
        # then: attribute question on a not-yet-dropped clue
        for i, clue in enumerate(self.clues):
            if i not in self.dropped and all(a.clue_index != i for a in self.answers):
                return attribute_template(clue, i, n)
        # finally: uncertainty again if anything remains
        if len(self.clues) >= 2 and asked_kinds.count("uncertainty") < 2:
            return uncertainty_question(self.clues, -1, n)
        return None

    # -- answering --------------------------------------------------
    def answer(self, value: str) -> None:
        if self.status != "asking" or self.pending is None:
            raise RuntimeError("no question pending")
        if len(self.answers) >= MAX_QUESTIONS:
            raise RuntimeError("question cap reached")
        q = self.pending
        self.answers.append(Answer(q.qid, q.kind, value, q.clue_index))

        if value != NOT_SURE:
            if q.kind == "uncertainty":
                # value is the doubted clue's text. Modifiers can be
                # dropped; anchors (main object/person) must be
                # replaced — searching without them destroys recall.
                for i, clue in enumerate(self.clues):
                    if clue.text == value:
                        if clue.kind in ANCHOR_KINDS:
                            self.doubted_anchor = i
                        else:
                            self._drop_clue(i)
                        break
            elif q.kind == "attribute" and q.clue_index is not None:
                self._replace_clue(q.clue_index, value)

        self.candidates = weighted_rerank(self.candidates, self.weights)
        self.pending = self._plan()
        if self.pending is None or len(self.answers) >= MAX_QUESTIONS:
            self.status = "done" if self.pending is None else "asking"

    def _drop_clue(self, i: int) -> None:
        self.dropped.add(i)
        self.weights[f"drop:{i}"] = W_DROPPED
        self.weights["original"] = min(self.weights.get("original", 1.0), 0.8)
        # Whole-corpus re-entry: revised query without the clue.
        rest = [c.text for j, c in enumerate(self.clues) if j not in self.dropped]
        if rest:
            hyp = Hypothesis("revised", "; ".join(rest), None)
            q = self.encoder.encode_texts([hyp.query_text])
            fresh = branch_union(self.index, [(hyp, q)], BRANCH_TOP_K)
            self._merge(fresh)

    def _replace_clue(self, i: int, value: str) -> None:
        parts = [
            value if j == i else c.text
            for j, c in enumerate(self.clues)
            if j not in self.dropped
        ]
        hyp = Hypothesis(f"answer:{i}", "; ".join(parts), None)
        q = self.encoder.encode_texts([hyp.query_text])
        fresh = branch_union(self.index, [(hyp, q)], BRANCH_TOP_K)
        self.weights[f"answer:{i}"] = W_DROPPED
        self._merge(fresh)

    def _merge(self, fresh: list[Candidate]) -> None:
        by_id = {c.image_id: c for c in self.candidates}
        for c in fresh:
            if c.image_id in by_id:
                old = by_id[c.image_id]
                for b, s in c.branch_scores.items():
                    old.branch_scores[b] = max(s, old.branch_scores.get(b, s))
                    if b not in old.branches:
                        old.branches.append(b)
                old.score = max(old.score, c.score)
            else:
                by_id[c.image_id] = c
        self.candidates = sorted(
            by_id.values(), key=lambda c: (-c.score, c.image_id)
        )

    # -- output -----------------------------------------------------
    def finish(self) -> dict:
        self.status = "done"
        top = self.candidates[:5]
        dropped_texts = [self.clues[i].text for i in sorted(self.dropped)]
        replaced = [a.value for a in self.answers
                    if a.kind == "attribute" and a.value != NOT_SURE]
        if dropped_texts:
            why = (f"After your answers we doubted: {', '.join(dropped_texts)}. "
                   "Results lean on the interpretations without them.")
        elif replaced:
            why = (f"You corrected a detail to: {', '.join(replaced)}. "
                   "Results lean on the corrected interpretation.")
        elif self.doubted_anchor is not None:
            why = (f"You doubted '{self.clues[self.doubted_anchor].text}' but we "
                   "didn't get a replacement, so the original ranking stands.")
        elif self.answers:
            why = ("Your answers didn't change anything (Not sure, or they "
                   "agreed with the original description); ranking unchanged.")
        else:
            why = "No follow-up answers; ranked by the original description and its relaxations."
        return {
            "session_id": self.session_id,
            "status": self.status,
            "questions_asked": len(self.answers),
            "dropped_clues": dropped_texts,
            "explanation": why,
            "results": [
                {"image_id": c.image_id, "score": round(c.score, 4),
                 "branches": c.branches}
                for c in top
            ],
        }

    def snapshot(self, top_k: int = 12) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "clues": [c.text for c in self.clues],
            "questions_asked": len(self.answers),
            "max_questions": MAX_QUESTIONS,
            "pending_question": (
                None if self.pending is None else {
                    "qid": self.pending.qid, "kind": self.pending.kind,
                    "text": self.pending.text, "options": self.pending.options,
                }
            ),
            "answers": [
                {"kind": a.kind, "value": a.value} for a in self.answers
            ],
            "results": [
                {"image_id": c.image_id, "score": round(c.score, 4),
                 "branches": c.branches}
                for c in self.candidates[:top_k]
            ],
        }
