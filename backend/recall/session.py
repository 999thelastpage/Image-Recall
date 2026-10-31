"""Recall search session: the ≤3-question repair loop.

State machine: new -> asking (<= MAX_QUESTIONS) -> done.
Answers never hard-filter. Dropping a clue raises the weight of
its leave-one-out branch and triggers whole-corpus re-entry with
the revised query; "Not sure" is neutral. Explanations are
templated from stored provenance (branches + answers), never
invented.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

import numpy as np

from .clues import Clue, Hypothesis, hypotheses
from .config import FINAL_TOP_K, MAX_QUESTIONS, branch_quota
from .questions import (
    CATEGORY_OPTIONS,
    COLOURS,
    NOT_SURE,
    SOMETHING_ELSE,
    YES_SURE,
    Question,
    attribute_template,
    category_question,
    details_question,
    dimension_question,
    replacement_question,
    subject_question,
    uncertainty_question,
)
from .rank import Candidate, branch_union, weighted_rerank
from .retrieve import ExactIndex

def _subjectless_score(candidate) -> float:
    """Best score from subject-less evidence only (revised/category/
    answer branches); falls back to the candidate's own score."""
    vals = [
        s for label, s in candidate.branch_scores.items()
        if label in ("revised", "category") or label.startswith("answer:")
    ]
    return max(vals) if vals else candidate.score


W_DROPPED = 1.6   # branch weight when user doubts that clue
W_CONFIRMED = 0.5  # branch weight for dropping a confirmed clue
ANCHOR_KINDS = {"object", "person"}  # doubted anchors get replaced, not dropped


def _swap_subject(text: str, old_ref: str, value: str,
                  _allow_last_word: bool = True) -> str | None:
    """Replace the subject word in ``text`` with ``value``.

    Tries, in order: ``old_ref``, it without a trailing 's', it plus
    's', it plus 'es', and (for the anchor phrase itself) the last
    word of ``text``. Returns None when nothing matches.
    """
    if not text or not value:
        return None
    old_ref = (old_ref or "").strip()
    candidates: list[str] = []
    if old_ref:
        candidates.append(old_ref)
        if old_ref.lower().endswith("s") and len(old_ref) > 1:
            candidates.append(old_ref[:-1])
        candidates.append(old_ref + "s")
        candidates.append(old_ref + "es")
    words = text.split()
    if _allow_last_word and words:
        candidates.append(words[-1])
    seen: set[str] = set()
    uniq: list[str] = []
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand.lower() in seen:
            continue
        seen.add(cand.lower())
        uniq.append(cand)
    for cand in uniq:
        pat = re.compile(r"\b" + re.escape(cand) + r"\b", re.I)
        m = pat.search(text)
        if m is None:
            continue
        matched = m.group(0)
        replacement = value
        if matched.lower().endswith("s"):
            if value.lower().endswith(("s", "x", "z", "ch", "sh")):
                replacement = value + "es"
            else:
                replacement = value + "s"
        if matched[:1].isupper() and replacement:
            replacement = replacement[:1].upper() + replacement[1:]
        return pat.sub(lambda _: replacement, text, count=1)
    return None


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
    subject_checked: bool = False
    dimension_checked: bool = False
    doubted_dimension: int | None = None
    question_history: list[str] = field(default_factory=list)
    needs_details: bool = False
    needs_category: bool = False
    status: str = "new"  # new | asking | done

    # -- setup ----------------------------------------------------
    def start(self) -> None:
        hyps = hypotheses(self.clues, substitution_colours=tuple(COLOURS),
                          max_substitutions=2)
        # One batched encode for all branches, not one call per branch.
        mat = self.encoder.encode_texts([h.query_text for h in hyps])
        queries = list(zip(hyps, mat, strict=True))
        self.candidates = branch_union(self.index, queries, branch_quota(len(self.index)))
        self.status = "asking"
        self.pending = self._plan()

    def _anchor_index(self) -> int | None:
        for i, clue in enumerate(self.clues):
            if clue.kind in ANCHOR_KINDS:
                return i
            if clue.kind in ("colour", "count") and clue.referent.strip():
                return i
        return None

    def _attribute_order(self) -> list[int]:
        """Eligible clue indices ordered by how much their drop
        branch splits the current top candidates (range of scores).
        Most discriminative first; stable fallback to clue order."""
        import statistics

        eligible = []
        for i, clue in enumerate(self.clues):
            if i in self.dropped:
                continue
            if i == self._anchor_index():
                continue  # anchor handled by subject/dimension, not re-asked
            if clue.kind not in ("colour", "count", "setting", "object", "person"):
                continue
            # Only an attribute answer settles a clue; the subject
            # or dimension check on the same index must not block it.
            if all(not (a.kind == "attribute" and a.clue_index == i)
                   for a in self.answers):
                eligible.append(i)
        top = self.candidates[:40]

        def spread(i: int) -> float:
            vals = [c.branch_scores[f"drop:{i}"] for c in top
                    if f"drop:{i}" in c.branch_scores]
            if len(vals) < 2:
                return 0.0
            return max(vals) - min(vals) + statistics.pstdev(vals)

        return sorted(eligible, key=lambda i: (-spread(i), i))

    def _uncertainty_clues(self) -> list[Clue]:
        picked = {a.value for a in self.answers if a.kind == "uncertainty"}
        pool = [c for j, c in enumerate(self.clues)
                if j not in self.dropped and c.text not in picked]
        if not self.subject_checked:
            return pool
        anchor = self._anchor_index()
        if anchor is None:
            return pool
        return [c for j, c in enumerate(self.clues)
                if j != anchor and j not in self.dropped
                and c.text not in picked]

    def _setting_options(self) -> list[str]:
        own = [c.text for j, c in enumerate(self.clues)
               if c.kind == "setting" and j not in self.dropped]
        generic = ["indoors", "outdoors (nature)", "outdoors (city)",
                   "at the beach / water", "on a kitchen table"]
        generic = [g for g in generic
                   if not any(g.split()[-1] in o for o in own)]
        out: list[str] = []
        for o in own + generic:
            if o not in out:
                out.append(o)
        return out

    def _object_options(self, exclude_index: int | None = None) -> list[str]:
        own = [c.referent for j, c in enumerate(self.clues)
               if c.referent.strip() and j != exclude_index
               and c.kind in ("object", "person", "colour", "count")]
        if exclude_index is not None:
            banned = self.clues[exclude_index].referent.strip().lower()
            own = [o for o in own if o.lower() != banned]
            fixed = ["dog", "cat", "car", "bus", "bicycle", "boat",
                     "train", "bird", "horse"]
            fixed = [o for o in fixed if o.lower() != banned]
        else:
            fixed = ["dog", "cat", "car", "bus", "bicycle", "boat",
                     "train", "bird", "horse"]
        out: list[str] = []
        for o in own + fixed:
            if o and o not in out:
                out.append(o)
        return out

    # -- planning ---------------------------------------------------
    def _plan(self) -> Question | None:
        q = self._plan_next()
        if q is None:
            return None
        sig = f"{q.kind}|{q.text}|{','.join(q.options)}"
        if sig in self.question_history:
            return None  # never ask the identical question twice
        self.question_history.append(sig)
        return q

    def _plan_next(self) -> Question | None:
        n = len(self.answers) + 1
        if n > MAX_QUESTIONS:
            return None
        asked_kinds = [a.kind for a in self.answers]
        if self.doubted_anchor is not None and not any(
            a.kind == "attribute" and a.clue_index == self.doubted_anchor
            for a in self.answers
        ):
            return replacement_question(
                self.clues[self.doubted_anchor], self.doubted_anchor, n,
                options=self._object_options(self.doubted_anchor))
        if self.needs_details and "details" not in asked_kinds:
            return details_question(n)
        if self.needs_category and "category" not in asked_kinds:
            return category_question(n)
        anchor = self._anchor_index()
        if anchor is not None and not self.subject_checked:
            return subject_question(self.clues[anchor], anchor, n)
        # a doubted bundled dimension gets its attribute next
        if self.doubted_dimension is not None and not any(
            a.kind == "attribute" and a.clue_index == self.doubted_dimension
            for a in self.answers
        ):
            i = self.doubted_dimension
            return attribute_template(self.clues[i], i, n)
        # bundled colour/count dimension, asked on its own so the
        # user can doubt the modifier without doubting the noun
        if anchor is not None and self.subject_checked \
                and not self.dimension_checked \
                and anchor not in self.dropped \
                and not any(a.kind == "attribute" and a.clue_index == anchor
                            for a in self.answers) \
                and self.clues[anchor].kind in ("colour", "count"):
            return dimension_question(self.clues[anchor], anchor, n)
        if "uncertainty" not in asked_kinds and len(self.clues) >= 2:
            pool = self._uncertainty_clues()
            if pool:
                return uncertainty_question(pool, -1, n)
        for i in self._attribute_order():
            clue = self.clues[i]
            if clue.kind == "setting":
                return attribute_template(clue, i, n,
                                          options=self._setting_options())
            if clue.kind in ("object", "person"):
                return attribute_template(clue, i, n,
                                          options=self._object_options())
            return attribute_template(clue, i, n)
        if len(self.clues) >= 2 and asked_kinds.count("uncertainty") < 2:
            pool = self._uncertainty_clues()
            if pool:
                return uncertainty_question(pool, -1, n)
        return None

    # -- answering --------------------------------------------------
    def answer(self, value: str) -> None:
        if self.status != "asking" or self.pending is None:
            raise RuntimeError("no question pending")
        if len(self.answers) >= MAX_QUESTIONS:
            raise RuntimeError("question cap reached")
        q = self.pending
        if q.kind != "details" and q.options and value not in q.options:
            raise RuntimeError(f"invalid answer {value!r} for this question")
        self.answers.append(Answer(q.qid, q.kind, value, q.clue_index))

        if q.kind == "subject":
            self.subject_checked = True
            anchor_idx = (
                q.clue_index if q.clue_index is not None
                else self._anchor_index()
            )
            if value == YES_SURE:
                pass  # no weight/drop change
            elif value == SOMETHING_ELSE:
                if anchor_idx is not None:
                    self.doubted_anchor = anchor_idx
            elif value == NOT_SURE:
                self.needs_category = True
                if anchor_idx is not None:
                    self._drop_clue(anchor_idx)
                    rest = [c for j, c in enumerate(self.clues)
                            if j not in self.dropped]
                    if not rest:
                        self.needs_details = True
        elif q.kind == "dimension":
            self.dimension_checked = True
            self.subject_checked = True
            dim_idx = q.clue_index if q.clue_index is not None \
                else self._anchor_index()
            if value == SOMETHING_ELSE:
                if dim_idx is not None:
                    self.doubted_dimension = dim_idx
            elif value == NOT_SURE:
                if dim_idx is not None:
                    # doubt only the modifier: boost its relax branch
                    self.weights[f"relax:{dim_idx}"] = W_DROPPED
        elif q.kind == "category":
            self.needs_category = False
            if value in CATEGORY_OPTIONS:
                term = value.split(" ", 1)[1].strip().lower() \
                    if " " in value else value.strip().lower()
                if term:
                    self._run_revised_search(extra_terms=[term],
                                             label="category")
        elif q.kind == "details":
            self.needs_details = False
            text = value.strip() if value else ""
            if text and text != NOT_SURE:
                self.clues.append(Clue(text, "other", ""))
                self.description = (
                    f"{self.description}; {text}" if self.description.strip()
                    else text
                )
                self._run_revised_search()
        elif value != NOT_SURE:
            if q.kind == "uncertainty":
                # value is the doubted clue's text. Modifiers can be
                # dropped; anchors (main object/person) must be
                # replaced: searching without them destroys recall.
                for i, clue in enumerate(self.clues):
                    if clue.text == value:
                        if i == self._anchor_index():
                            self.doubted_anchor = i
                        else:
                            self._drop_clue(i)
                        break
            elif q.kind == "attribute" and q.clue_index is not None:
                if self.doubted_anchor is not None \
                        and q.clue_index == self.doubted_anchor:
                    effective = self._replace_clue(
                        q.clue_index, value, subject=True)
                    self.answers[-1].value = effective
                else:
                    self._replace_clue(q.clue_index, value)

        self.candidates = weighted_rerank(self.candidates, self.weights)
        anchor = self._anchor_index()
        if anchor is not None and anchor in self.dropped:
            self.candidates = sorted(
                self.candidates,
                key=lambda c: (-_subjectless_score(c), -c.score, c.image_id),
            )
        self.pending = self._plan()
        if self.pending is None or len(self.answers) >= MAX_QUESTIONS:
            self.status = "done" if self.pending is None else "asking"

    def _rescore_all(self, label: str, query) -> None:
        """Give EVERY candidate a score under the revised query.

        Branch re-entry only merges the revised query's top-k; a
        candidate outside that cut used to get no score from the
        boosted branch and sank when others were weighted up. One
        matvec over the union removes that blindness."""
        pos = {iid: r for r, iid in enumerate(self.index.image_ids)}
        rows = [pos[c.image_id] for c in self.candidates
                if c.image_id in pos]
        cands = [c for c in self.candidates if c.image_id in pos]
        if not rows:
            return
        scores = self.index.score_rows(query, rows)
        for c, s in zip(cands, scores, strict=True):
            c.branch_scores[label] = max(float(s),
                                         c.branch_scores.get(label, float(s)))
            if label not in c.branches:
                c.branches.append(label)

    def _run_revised_search(self, extra_terms: list[str] | None = None,
                            label: str = "revised") -> None:
        """Whole-corpus re-entry with the subject-less query (all
        non-dropped clues), then rescore every candidate under it."""
        anchor = self._anchor_index()
        if anchor is not None and anchor in self.dropped:
            anchor_clue = self.clues[anchor]
            words = anchor_clue.text.split()
            old_ref = anchor_clue.referent.strip() or (words[-1] if words else "")
            if old_ref:
                pat = re.compile(r"\b" + re.escape(old_ref) + r"\b", re.I)
                rest = []
                for j, c in enumerate(self.clues):
                    if j in self.dropped:
                        continue
                    t = " ".join(pat.sub("", c.text).split()).strip(" ;,")
                    if len(t) >= 2:
                        rest.append(t)
                remainder = " ".join(
                    pat.sub("", anchor_clue.text).split()).strip(" ;,")
                if len(remainder) >= 2:
                    rest.append(remainder)
            else:
                rest = [c.text for j, c in enumerate(self.clues)
                        if j not in self.dropped]
        else:
            rest = [c.text for j, c in enumerate(self.clues)
                    if j not in self.dropped]
        if extra_terms:
            rest = rest + [t for t in extra_terms if t]
        if not rest:
            return
        hyp = Hypothesis(label, "; ".join(rest), None)
        q = self.encoder.encode_texts([hyp.query_text])
        fresh = branch_union(self.index, [(hyp, q)], branch_quota(len(self.index)))
        self._merge(fresh)
        self.weights[label] = W_DROPPED
        self._rescore_all(label, q)

    def _drop_clue(self, i: int) -> None:
        self.dropped.add(i)
        self.weights[f"drop:{i}"] = W_DROPPED
        self.weights["original"] = min(self.weights.get("original", 1.0), 0.8)
        # Whole-corpus re-entry: revised query without the clue.
        self._run_revised_search()

    def _replace_clue(self, i: int, value: str, subject: bool = False) -> str:
        if subject:
            clue = self.clues[i]
            words = clue.text.split()
            old_ref = clue.referent.strip() or (words[-1] if words else "")
            parts = []
            effective = value
            for j, c in enumerate(self.clues):
                if j in self.dropped:
                    continue
                # The last-word fallback is only safe on the anchor
                # phrase itself; other clues keep their text when the
                # subject word does not appear in them (e.g. 'together').
                part = _swap_subject(c.text, old_ref, value,
                                     _allow_last_word=(j == i)) or (
                    value if j == i else c.text)
                if j == i:
                    effective = part
                parts.append(part)
        else:
            clue0 = self.clues[i]
            if clue0.kind == "colour" and value.lower() in COLOURS \
                    and clue0.referent.strip():
                effective = f"{value} {clue0.referent}"
            elif clue0.kind == "count" and value[:1].isdigit() \
                    and clue0.referent.strip():
                effective = f"{value} {clue0.referent}"
            else:
                effective = value
            parts = [
                effective if j == i else c.text
                for j, c in enumerate(self.clues)
                if j not in self.dropped
            ]
        hyp = Hypothesis(f"answer:{i}", "; ".join(parts), None)
        q = self.encoder.encode_texts([hyp.query_text])
        fresh = branch_union(self.index, [(hyp, q)], branch_quota(len(self.index)))
        self.weights[f"answer:{i}"] = W_DROPPED
        self._merge(fresh)
        self._rescore_all(hyp.label, q)
        return effective

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
                old.fused += c.fused
            else:
                by_id[c.image_id] = c
        self.candidates = sorted(
            by_id.values(), key=lambda c: (-c.score, c.image_id)
        )

    # -- persistence --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "description": self.description,
            "clues": [{"text": c.text, "kind": c.kind, "referent": c.referent}
                      for c in self.clues],
            "candidates": [
                {"image_id": c.image_id, "score": c.score,
                 "branches": c.branches, "branch_scores": c.branch_scores,
                 "fused": c.fused}
                for c in self.candidates
            ],
            "weights": self.weights,
            "answers": [{"qid": a.qid, "kind": a.kind, "value": a.value,
                         "clue_index": a.clue_index} for a in self.answers],
            "dropped": sorted(self.dropped),
            "doubted_anchor": self.doubted_anchor,
            "subject_checked": self.subject_checked,
            "dimension_checked": self.dimension_checked,
            "doubted_dimension": self.doubted_dimension,
            "question_history": list(self.question_history),
            "needs_details": self.needs_details,
            "needs_category": self.needs_category,
            "status": self.status,
            "pending": None if self.pending is None else {
                "qid": self.pending.qid, "kind": self.pending.kind,
                "text": self.pending.text, "options": self.pending.options,
                "clue_index": self.pending.clue_index,
            },
        }

    @classmethod
    def from_dict(cls, index: ExactIndex, encoder, data: dict) -> "SearchSession":
        s = cls(
            index=index, encoder=encoder,
            description=data["description"],
            clues=[Clue(c["text"], c.get("kind", "other"), c.get("referent", ""))
                   for c in data["clues"]],
            session_id=data["session_id"],
        )
        s.candidates = [Candidate(c["image_id"], c["score"], c["branches"],
                                  c.get("branch_scores", {}), c.get("fused", 0.0))
                        for c in data["candidates"]]
        s.weights = dict(data.get("weights", {}))
        s.answers = [Answer(a["qid"], a["kind"], a["value"], a.get("clue_index"))
                     for a in data.get("answers", [])]
        s.dropped = set(data.get("dropped", []))
        s.doubted_anchor = data.get("doubted_anchor")
        s.subject_checked = bool(data.get(
            "subject_checked",
            any(a.kind == "subject" for a in s.answers),
        ))
        s.dimension_checked = bool(data.get(
            "dimension_checked",
            any(a.kind == "dimension" for a in s.answers)))
        s.doubted_dimension = data.get("doubted_dimension")
        s.question_history = list(data.get("question_history", []))
        s.needs_details = bool(data.get("needs_details", False))
        if "needs_category" in data:
            s.needs_category = bool(data["needs_category"])
        else:
            # Old sessions: owed if subject was doubted and no
            # category answer has been recorded yet.
            s.needs_category = bool(
                any(a.kind == "subject" and a.value == NOT_SURE
                    for a in s.answers)
                and not any(a.kind == "category" for a in s.answers)
            )
        s.status = data.get("status", "asking")
        p = data.get("pending")
        s.pending = None if p is None else Question(
            qid=p["qid"], kind=p["kind"], text=p["text"],
            options=p["options"], clue_index=p.get("clue_index"))
        return s

    # -- output -----------------------------------------------------
    def finish(self) -> dict:
        self.status = "done"
        top = self.candidates[:FINAL_TOP_K]
        dropped_texts = [self.clues[i].text for i in sorted(self.dropped)]
        replaced = [
            f"'{self.clues[a.clue_index].text}' → '{a.value}'"
            if a.clue_index is not None else f"'{a.value}'"
            for a in self.answers
            if a.kind == "attribute" and a.value != NOT_SURE
        ]
        if dropped_texts:
            why = (f"After your answers we doubted: {', '.join(dropped_texts)}. "
                   "Results lean on the interpretations without them.")
        elif replaced:
            why = (f"You corrected: {'; '.join(replaced)}. "
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
