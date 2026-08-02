"""Packing fixed-length sequences from variable-length documents (`[01:11:55]`).

Two policies, chosen by data type (`[02:24:06]`):
  * "concat"     — pretraining lanes: concatenate docs with EOS boundaries, chopping
                   the doc that crosses the edge. High utilisation.
  * "structured" — SFT / agentic lanes: one document per sequence, never leaking
                   unrelated examples together; loss only on answer spans.

Each packed sequence carries the three parallel arrays the loader must expose
(`[00:48:34]`): a **loss mask** (next-token targets that bear loss), **segment ids**
(block-diagonal attention — no attention across an EOS boundary `[01:03:46]`), and
**position ids** (reset per document).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

# a "doc" from the provider: {"ids": np.uint32, "loss": np.bool_, "shard_id", "doc_id"}
DocProvider = Callable[[], Optional[dict]]


def _finalize(toks, loss_tok, seg, pos, seq_len, pad_id) -> dict:
    # pad to seq_len
    n_real = len(toks)
    while len(toks) < seq_len:
        toks.append(pad_id)
        loss_tok.append(False)
        seg.append(-1)
        pos.append(0)
    toks = np.asarray(toks[:seq_len], dtype=np.uint32)
    loss_tok = np.asarray(loss_tok[:seq_len], dtype=bool)
    seg = np.asarray(seg[:seq_len], dtype=np.int32)
    pos = np.asarray(pos[:seq_len], dtype=np.int32)
    # next-token loss mask: position i trains to predict i+1, iff same segment,
    # target is loss-bearing, and target is not pad
    loss_mask = np.zeros(seq_len, dtype=bool)
    same_seg = (seg[:-1] == seg[1:]) & (seg[:-1] >= 0)
    tgt_loss = loss_tok[1:]
    loss_mask[:-1] = same_seg & tgt_loss
    return {"tokens": toks, "loss_mask": loss_mask, "segment_ids": seg,
            "position_ids": pos, "n_real": int(n_real),
            "n_loss": int(loss_mask.sum())}


def fill_sequence(provider: DocProvider, seq_len: int, policy: str,
                  eos_id: int, pad_id: int) -> Optional[dict]:
    """Fill exactly ONE packed sequence, drawing documents from `provider`.
    Returns None only when the provider is exhausted before any content."""
    toks: List[int] = []
    loss_tok: List[bool] = []
    seg: List[int] = []
    pos: List[int] = []
    members: List[dict] = []

    if policy == "structured":
        doc = provider()
        if doc is None:
            return None
        ids = doc["ids"].tolist()[: seq_len - 1]
        lf = doc["loss"].tolist()[: len(ids)]
        for k, (tid, lb) in enumerate(zip(ids, lf)):
            toks.append(int(tid)); loss_tok.append(bool(lb)); seg.append(0); pos.append(k)
        # EOS closes the trajectory; not an answer token -> no loss
        toks.append(eos_id); loss_tok.append(False); seg.append(0); pos.append(len(ids))
        members.append({"shard_id": doc["shard_id"], "doc_id": doc["doc_id"],
                        "n_tokens": len(ids)})
        out = _finalize(toks, loss_tok, seg, pos, seq_len, pad_id)
        out["members"] = members
        out["policy"] = policy
        return out

    # ---- concat (pretraining) ----
    seg_idx = 0
    while len(toks) < seq_len:
        doc = provider()
        if doc is None:
            break
        ids = doc["ids"].tolist()
        lf = doc["loss"].tolist()
        placed = 0
        for k, (tid, lb) in enumerate(zip(ids, lf)):
            if len(toks) >= seq_len:
                break
            toks.append(int(tid)); loss_tok.append(bool(lb))
            seg.append(seg_idx); pos.append(k)
            placed += 1
        members.append({"shard_id": doc["shard_id"], "doc_id": doc["doc_id"],
                        "n_tokens": placed, "chopped": placed < len(ids)})
        # EOS boundary if the doc fit whole and there is room
        if placed == len(ids) and len(toks) < seq_len:
            toks.append(eos_id); loss_tok.append(True)
            seg.append(seg_idx); pos.append(placed)
        seg_idx += 1
    if not toks:
        return None
    out = _finalize(toks, loss_tok, seg, pos, seq_len, pad_id)
    out["members"] = members
    out["policy"] = policy
    return out


def policy_for_lane(lane: str) -> str:
    return "structured" if lane in ("agentic", "reasoning") else "concat"


def batch_utilization(seqs: List[dict], seq_len: int) -> dict:
    cap = len(seqs) * seq_len
    real = sum(s["n_real"] for s in seqs)
    loss = sum(s["n_loss"] for s in seqs)
    return {"capacity_tokens": cap, "real_tokens": real, "loss_tokens": loss,
            "packing_utilization": round(real / cap, 4) if cap else 0.0,
            "loss_bearing_fraction": round(loss / cap, 4) if cap else 0.0}
