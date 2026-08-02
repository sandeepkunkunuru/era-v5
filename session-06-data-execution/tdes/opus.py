"""OPUS: in-training data selection (`[00:04:05]`, `[01:49:26]`). A candidate is
scored on its **initial tokens** (the proxy "has also gone through 512" `[00:05:06]`);
the score decides accept / reject / defer. Because the proxy is English/code-biased,
Indic and agentic score low — so the stream applies a **protected-floor override**
to keep them present (the reason the always-on floor exists).

Scoring is a PURE function of (master_seed, shard_id, doc_idx, token prefix), so the
whole decision trail replays exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .hashing import derive_seed

# proxy alignment per lane: high for English/code/math, low for Indic/agentic
_ALIGN: Dict[str, float] = {
    "general_web": 0.70, "code": 0.72, "math": 0.68,
    "reasoning": 0.55, "agentic": 0.28, "indic": 0.25,
}
ACCEPT_AT = 0.50
DEFER_AT = 0.36
PREFIX = 16   # "initial 512 tokens" scaled down to our tiny seq_len


@dataclass
class OpusConfig:
    accept_at: float = ACCEPT_AT
    defer_at: float = DEFER_AT
    prefix: int = PREFIX


class Opus:
    def __init__(self, master_seed: int, cfg: OpusConfig = None):
        self.master_seed = master_seed
        self.cfg = cfg or OpusConfig()

    def score(self, shard_id: str, doc_idx: int, prefix_ids) -> float:
        lane = shard_id.split(".")[0]
        base = _ALIGN.get(lane, 0.5)
        pref = tuple(int(x) for x in prefix_ids[: self.cfg.prefix])
        s = derive_seed(self.master_seed, "opus", shard_id, doc_idx, pref)
        noise = (s % 10_000) / 10_000.0 * 0.4 - 0.2   # U(-0.2, 0.2)
        return max(0.0, min(1.0, base + noise))

    def decide(self, candidate_id: str, shard_id: str, doc_idx: int,
               prefix_ids, stage: str, step: int, doc_id: str = None) -> dict:
        sc = self.score(shard_id, doc_idx, prefix_ids)
        if sc >= self.cfg.accept_at:
            decision = "accept"
        elif sc >= self.cfg.defer_at:
            decision = "defer"
        else:
            decision = "reject"
        # doc_id is carried so the trail joins DIRECTLY to the consumption ledger's
        # sequence members — i.e. "why was this document consumed?" is answerable.
        return {"candidate_id": candidate_id, "shard_id": shard_id,
                "doc_idx": doc_idx, "doc_id": doc_id, "lane": shard_id.split(".")[0],
                "stage": stage, "step": step, "score": round(sc, 4),
                "decision": decision, "protected_override": False,
                "reason": f"score {sc:.3f} vs accept {self.cfg.accept_at}"}


def summarize(records: List[dict]) -> dict:
    from collections import Counter
    dec = Counter(r["decision"] for r in records)
    overrides = sum(1 for r in records if r["protected_override"])
    by_lane = {}
    for r in records:
        d = by_lane.setdefault(r["lane"], Counter())
        d[r["decision"]] += 1
        if r["protected_override"]:
            d["override"] += 1
    return {"total": len(records), "by_decision": dict(dec),
            "protected_overrides": overrides,
            "by_lane": {k: dict(v) for k, v in by_lane.items()}}
