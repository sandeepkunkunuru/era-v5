"""The deterministic batch stream — the piece everything else proves things about.

`SamplerState` (step + per-lane cursor/epoch + deferral queue) is small and fully
serialisable. `Stream.next_batch(state)` is a PURE function of (shard set, compiled
schedule, OPUS, state): no model, no wall-clock, no global RNG. So:

  * resume  = restore state -> next_batch gives the identical next batch
  * replay  = restore an earlier state -> re-run -> identical batch hashes
  * fork    = restore a state, change seed/schedule -> a valid divergent stream

Protected lanes (Indic/agentic/reasoning floors) bypass OPUS rejection — the
protected-floor override (`[01:34:14]` OPUS + floors), logged in the audit trail.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hashing import canonical, derive_seed, sha256_bytes, sha256_json
from .opus import Opus
from .packing import batch_utilization, fill_sequence, policy_for_lane
from .schedule import Schedule
from .shards import ShardSet

MAX_ATTEMPTS = 6   # bound OPUS re-draws for non-protected lanes -> guaranteed termination


@dataclass
class SamplerState:
    t: int = 0
    cursors: Dict[str, dict] = field(default_factory=dict)   # lane -> {epoch,pos}
    deferred: List[list] = field(default_factory=list)       # [[shard_id, doc_idx], ...]

    def to_dict(self) -> dict:
        return {"t": self.t, "cursors": self.cursors, "deferred": self.deferred}

    @classmethod
    def from_dict(cls, d: dict) -> "SamplerState":
        return cls(t=d["t"], cursors=copy.deepcopy(d["cursors"]),
                   deferred=[list(x) for x in d["deferred"]])

    def hash(self) -> str:
        return sha256_json(self.to_dict())


class Stream:
    def __init__(self, shards: ShardSet, schedule: Schedule, opus: Opus,
                 eos_id: int, pad_id: int, master_seed: int):
        self.shards = shards
        self.schedule = schedule
        self.opus = opus
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.master_seed = master_seed
        # flat, deterministic per-lane document lists over TRAINABLE shards only
        self.lane_docs: Dict[str, List[Tuple[str, int]]] = {}
        for s in sorted(shards.trainable(), key=lambda s: s.shard_id):
            for i in range(len(s.docs)):
                self.lane_docs.setdefault(s.lane, []).append((s.shard_id, i))
        self._perm_cache: Dict[Tuple[str, int], np.ndarray] = {}

    # ---- lane ordering ------------------------------------------------------
    def _perm(self, lane: str, epoch: int) -> np.ndarray:
        key = (lane, epoch)
        if key not in self._perm_cache:
            n = len(self.lane_docs.get(lane, []))
            rng = np.random.default_rng(derive_seed(self.master_seed, "perm", lane, epoch))
            self._perm_cache[key] = rng.permutation(n) if n else np.array([], dtype=int)
        return self._perm_cache[key]

    def initial_state(self) -> SamplerState:
        cursors = {ln: {"epoch": 0, "pos": 0} for ln in self.lane_docs}
        return SamplerState(t=0, cursors=cursors, deferred=[])

    def _draw_raw(self, lane: str, state: SamplerState) -> Optional[Tuple[str, int]]:
        docs = self.lane_docs.get(lane, [])
        if not docs:
            return None
        cur = state.cursors[lane]
        if cur["pos"] >= len(docs):
            cur["epoch"] += 1
            cur["pos"] = 0
        perm = self._perm(lane, cur["epoch"])
        idx = int(perm[cur["pos"]])
        cur["pos"] += 1
        return docs[idx]

    def _load(self, shard_id: str, doc_idx: int) -> dict:
        s = self.shards.by_id[shard_id]
        ids, loss = s.doc_arrays(doc_idx)
        return {"ids": np.asarray(ids), "loss": np.asarray(loss),
                "shard_id": shard_id, "doc_id": s.docs[doc_idx]["doc_id"]}

    # ---- one batch ----------------------------------------------------------
    def next_batch(self, state: SamplerState) -> Tuple[dict, SamplerState, dict, List[dict]]:
        state = SamplerState.from_dict(state.to_dict())   # never mutate caller's state
        state_before = SamplerState.from_dict(state.to_dict())
        t = state.t
        stage = self.schedule.stage_at(t)
        seq_len = stage["seq_len"]
        alloc = self.schedule.lane_alloc(t)
        floors = stage["floors"]
        opus_records: List[dict] = []
        sequences: List[dict] = []
        seq_meta: List[dict] = []
        actual_counts: Dict[str, int] = {ln: 0 for ln in alloc}

        for lane in sorted(alloc):
            need = alloc[lane]
            if need == 0:
                continue
            protected = lane in floors
            policy = policy_for_lane(lane)

            def provider():
                # returns an accepted doc, applying OPUS accept/reject/defer/override
                for _attempt in range(MAX_ATTEMPTS):
                    drawn = self._draw_raw(lane, state)
                    if drawn is None:
                        return None
                    shard_id, doc_idx = drawn
                    doc = self._load(shard_id, doc_idx)
                    cand_id = f"{t}:{shard_id}:{doc_idx}"
                    rec = self.opus.decide(cand_id, shard_id, doc_idx,
                                           doc["ids"], stage["name"], t,
                                           doc_id=doc["doc_id"])
                    if rec["decision"] == "accept":
                        opus_records.append(rec)
                        return doc
                    if protected:
                        rec["protected_override"] = True
                        rec["reason"] += " | protected-floor override"
                        opus_records.append(rec)
                        return doc
                    if rec["decision"] == "defer":
                        state.deferred.append([shard_id, doc_idx])
                    opus_records.append(rec)
                    # rejected/deferred -> draw again
                # exhausted attempts: accept the last draw to guarantee progress
                rec["decision"] = "accept_after_max_attempts"
                opus_records.append(rec)
                return doc

            produced = 0
            while produced < need:
                seq = fill_sequence(provider, seq_len, policy, self.eos_id, self.pad_id)
                if seq is None:
                    break
                seq["lane"] = lane
                sequences.append(seq)
                seq_meta.append({"lane": lane, "policy": seq["policy"],
                                 "members": seq["members"], "n_real": seq["n_real"],
                                 "n_loss": seq["n_loss"]})
                produced += 1
                actual_counts[lane] += 1

        # ---- assemble batch arrays ----
        B = len(sequences)
        tokens = np.stack([s["tokens"] for s in sequences]) if B else np.zeros((0, seq_len), np.uint32)
        loss_mask = np.stack([s["loss_mask"] for s in sequences]) if B else np.zeros((0, seq_len), bool)
        seg = np.stack([s["segment_ids"] for s in sequences]) if B else np.zeros((0, seq_len), np.int32)
        pos = np.stack([s["position_ids"] for s in sequences]) if B else np.zeros((0, seq_len), np.int32)

        payload = tokens.tobytes() + loss_mask.tobytes() + seg.tobytes() + pos.tobytes()
        batch_sha = sha256_bytes(payload)
        batch_id = f"b{t:06d}-{sha256_bytes(canonical([self.master_seed, t]))[:8]}"
        util = batch_utilization(sequences, seq_len)

        state.t = t + 1
        batch = {"batch_id": batch_id, "batch_sha256": batch_sha, "step": t,
                 "stage": stage["name"], "seq_len": seq_len,
                 "tokens": tokens, "loss_mask": loss_mask, "segment_ids": seg,
                 "position_ids": pos, "utilization": util,
                 "lane_counts": actual_counts}
        record = {"step": t, "stage": stage["name"], "seq_len": seq_len,
                  "batch_id": batch_id, "batch_sha256": batch_sha,
                  "planned_alloc": alloc, "actual_lane_counts": actual_counts,
                  "utilization": util, "n_sequences": B,
                  "sequences": seq_meta,
                  "state_before": state_before.to_dict(),
                  "state_before_hash": state_before.hash()}
        return batch, state, record, opus_records
