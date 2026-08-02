"""A tiny, deterministic, multi-lane corpus. "Keep your dataset small — don't make
me download a terabyte" (`[02:26:07]`). Text is templated (seeded), so the corpus
is byte-identical on every run.

Lanes mirror the Session 5 plan (general_web, code, math, reasoning, agentic, indic).
Two lanes are *structured* (reasoning, agentic): each carries context spans (masked,
no loss) and answer spans (loss-bearing) — so loss masks are real, not cosmetic
(`[00:54:39]`, agentic tool observations are context, `[01:23:54]`).

Some documents are marked split="eval": the evaluation firewall must keep them out
of every loss-bearing batch (`[02:24:06]`).
"""
from __future__ import annotations

import random
from typing import List, Optional, TypedDict

LANES = ["general_web", "code", "math", "reasoning", "agentic", "indic"]


class Segment(TypedDict):
    text: str
    loss: bool          # do answer tokens in this segment bear loss?


class Document(TypedDict):
    doc_id: str
    lane: str
    split: str                       # "train" | "eval"
    text: str                        # flat text (for unstructured lanes / hashing)
    segments: Optional[List[Segment]]  # set for structured lanes, else None


_WEB = ("the river valley trade route connected distant towns and markets . "
        "farmers recorded the monsoon calendar to plan the sowing season . "
        "a public library opened near the old railway station last spring . "
        "historians debate how the ancient port city managed its grain supply .").split(" . ")
_CODE = ["def add(a, b):\n    return a + b",
         "for i in range(n):\n    total += weights[i] * x[i]",
         "class Node:\n    def __init__(self, v):\n        self.v = v\n        self.next = None",
         "with open(path) as f:\n    data = json.load(f)"]
_MATHQ = ["Compute the sum of integers from 1 to {n}.",
          "How many multiples of 3 are there below {n}?",
          "Evaluate {a} times {b} minus {c}."]
_INDIC = ["गाँव के पास एक पुराना पुस्तकालय खुला है ।",
          "किसान मानसून के अनुसार बुवाई की योजना बनाते हैं ।",
          "నది లోయ వాణిజ్య మార్గం దూర పట్టణాలను కలిపింది ।",
          "பழைய துறைமுக நகரம் தானிய விநியோகத்தை நிர்வகித்தது ।"]


def _mk(doc_id, lane, split, text, segments=None) -> Document:
    return {"doc_id": doc_id, "lane": lane, "split": split, "text": text,
            "segments": segments}


def build_corpus(seed: int = 6006, eval_fraction: float = 0.12) -> List[Document]:
    rng = random.Random(seed)
    docs: List[Document] = []

    def maybe_eval(i: int) -> str:
        # deterministic: every ~1/eval_fraction-th doc is held out for evaluation
        return "eval" if (i % max(2, round(1 / eval_fraction)) == 0) else "train"

    # ---- unstructured pretraining lanes ----
    for i in range(46):
        n = rng.randint(2, 5)
        text = " . ".join(rng.choice(_WEB) for _ in range(n)) + " ."
        docs.append(_mk(f"web-{i:03d}", "general_web", maybe_eval(i), text))
    for i in range(24):
        text = "\n\n".join(rng.choice(_CODE) for _ in range(rng.randint(1, 3)))
        docs.append(_mk(f"code-{i:03d}", "code", maybe_eval(i), text))
    for i in range(24):
        q = rng.choice(_MATHQ).format(n=rng.randint(10, 99), a=rng.randint(2, 12),
                                      b=rng.randint(2, 12), c=rng.randint(1, 9))
        docs.append(_mk(f"math-{i:03d}", "math", maybe_eval(i), q))
    for i in range(22):
        text = " ".join(rng.choice(_INDIC) for _ in range(rng.randint(1, 3)))
        docs.append(_mk(f"indic-{i:03d}", "indic", maybe_eval(i), text))

    # ---- structured lanes: reasoning (problem->solution) ----
    for i in range(18):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        problem = f"Problem: what is {a} plus {b}? Think step by step."
        solution = f"Solution: {a} plus {b} equals {a + b}. Answer: {a + b}."
        segs: List[Segment] = [{"text": problem, "loss": False},
                               {"text": " " + solution, "loss": True}]
        docs.append(_mk(f"reason-{i:03d}", "reasoning", maybe_eval(i),
                        problem + " " + solution, segs))

    # ---- structured lanes: agentic (user + tool obs = context; assistant = loss) ----
    for i in range(14):
        user = f"User: find the population rank of city_{rng.randint(1, 40)}."
        plan = " Assistant: I will call the search tool with the city name."
        obs = " Observation: {\"rank\": %d, \"source\": \"gazetteer\"}." % rng.randint(1, 50)
        final = " Assistant: The rank is recorded; reporting it as the final answer."
        segs = [{"text": user, "loss": False}, {"text": plan, "loss": True},
                {"text": obs, "loss": False}, {"text": final, "loss": True}]
        docs.append(_mk(f"agentic-{i:03d}", "agentic", maybe_eval(i),
                        user + plan + obs + final, segs))

    return docs


def corpus_stats(docs: List[Document]) -> dict:
    from collections import Counter
    by_lane = Counter(d["lane"] for d in docs)
    by_split = Counter(d["split"] for d in docs)
    return {"n_docs": len(docs), "by_lane": dict(by_lane), "by_split": dict(by_split)}
