"""Compile the Session-5 mixture (a human recipe) into a machine-followable
schedule (`[01:31:12]`): curriculum stages, per-lane weights, protected floors,
and a feasibility check ("you want 38 billion tokens … you don't have [the]
dataset, so it will fail" `[01:34:14]`).

`lane_alloc(t, B)` is a PURE function of (master_seed, t, B): the number of
sequences of each lane in the batch at step t, with protected floors always met.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .corpus import LANES
from .hashing import derive_seed

# Default curriculum: general -> mid -> anneal. Weights per lane (sum ~1), plus
# protected floors that the allocator may never cross (Indic/agentic/reasoning).
DEFAULT_STAGES: List[dict] = [
    {"name": "seed", "steps": 10, "seq_len": 128,
     "weights": {"general_web": .55, "code": .12, "math": .10,
                 "reasoning": .06, "agentic": .05, "indic": .12},
     "floors": {"indic": .10, "agentic": .05, "reasoning": .05}},
    {"name": "mid", "steps": 15, "seq_len": 128,
     "weights": {"general_web": .38, "code": .20, "math": .15,
                 "reasoning": .10, "agentic": .05, "indic": .12},
     "floors": {"indic": .10, "agentic": .05, "reasoning": .05}},
    {"name": "anneal", "steps": 15, "seq_len": 128,
     "weights": {"general_web": .20, "code": .20, "math": .15,
                 "reasoning": .15, "agentic": .12, "indic": .18},
     "floors": {"indic": .12, "agentic": .08, "reasoning": .08}},
]


def _largest_remainder(weights: Dict[str, float], total: int) -> Dict[str, int]:
    raw = {k: w * total for k, w in weights.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = total - sum(base.values())
    order = sorted(weights, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    return base


@dataclass
class Schedule:
    stages: List[dict]
    master_seed: int
    batch_size: int

    def __post_init__(self):
        self.total_steps = sum(s["steps"] for s in self.stages)
        self._bounds = []
        acc = 0
        for s in self.stages:
            self._bounds.append((acc, acc + s["steps"], s))
            acc += s["steps"]

    def stage_at(self, t: int) -> dict:
        for lo, hi, s in self._bounds:
            if lo <= t < hi:
                return s
        return self._bounds[-1][2]

    def seq_len(self, t: int) -> int:
        return self.stage_at(t)["seq_len"]

    def lane_alloc(self, t: int) -> Dict[str, int]:
        """Deterministic per-lane sequence counts for the batch at step t, floors met."""
        stage = self.stage_at(t)
        B = self.batch_size
        weights = {ln: stage["weights"].get(ln, 0.0) for ln in LANES}
        alloc = _largest_remainder(weights, B)
        floors = {ln: math.ceil(fr * B) for ln, fr in stage["floors"].items()}

        # bump floor lanes up to their minimum
        deficit = 0
        for ln, fmin in floors.items():
            if alloc[ln] < fmin:
                deficit += fmin - alloc[ln]
                alloc[ln] = fmin
        # reclaim `deficit` from the largest reducible lanes (never below a floor)
        rng_seed = derive_seed(self.master_seed, "alloc", t)
        while deficit > 0:
            reducible = [ln for ln in LANES
                         if alloc[ln] > floors.get(ln, 0)]
            if not reducible:
                break
            # largest first; deterministic tie-break by hashed name+seed
            ln = max(reducible, key=lambda k: (alloc[k],
                                               derive_seed(rng_seed, k)))
            alloc[ln] -= 1
            deficit -= 1
        return alloc

    def planned_shares(self) -> Dict[str, float]:
        """Average planned lane share across the whole run (for compliance audit)."""
        tot = {ln: 0 for ln in LANES}
        for t in range(self.total_steps):
            for ln, c in self.lane_alloc(t).items():
                tot[ln] += c
        denom = self.total_steps * self.batch_size
        return {ln: tot[ln] / denom for ln in LANES}


def compile_schedule(stages: List[dict], master_seed: int, batch_size: int) -> Schedule:
    return Schedule(stages, master_seed, batch_size)


def feasibility(schedule: Schedule, tokens_per_lane: Dict[str, int],
                max_repeat: float = 4.0) -> dict:
    """Planned tokens per lane over the run vs available unique tokens. A lane whose
    demand exceeds available*max_repeat is INFEASIBLE (the lecture's 'it will fail')."""
    seq_len = schedule.stages[0]["seq_len"]
    planned = schedule.planned_shares()
    total_seqs = schedule.total_steps * schedule.batch_size
    report = {"max_repeat": max_repeat, "lanes": {}, "feasible": True}
    for ln, share in planned.items():
        demand = int(share * total_seqs * seq_len)
        avail = tokens_per_lane.get(ln, 0)
        rep = demand / avail if avail else float("inf")
        ok = rep <= max_repeat
        report["lanes"][ln] = {"demand_tokens": demand, "available_tokens": avail,
                               "repeat_factor": round(rep, 2), "feasible": ok}
        if not ok:
            report["feasible"] = False
    return report
