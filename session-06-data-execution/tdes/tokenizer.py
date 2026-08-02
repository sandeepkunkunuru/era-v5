"""A small, frozen, byte-level BPE tokenizer — self-contained (no network, no
external deps) so `python run_demo.py` runs anywhere. Session 6: tokenization is
done once, *before* training, and **frozen**; its identity travels as a content
hash and is verified before any shard is trusted (`[01:26:09]`).

Determinism: training is greedy with deterministic tie-breaking, so the same
corpus always yields byte-identical merges -> the same `tokenizer_hash`.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .hashing import sha256_json

# specials live at the TOP of the id space (after byte-base + merges is fixed at
# freeze time we instead reserve them at fixed low-ish ids computed from vocab).
SPECIALS = ["<pad>", "<bos>", "<eos>"]
_PRETOK = re.compile(r"\s*\S+|\s+")


def _pretokenize(text: str) -> List[bytes]:
    return [m.group(0).encode("utf-8") for m in _PRETOK.finditer(text)]


def _get_pairs(seq: Tuple[int, ...]) -> Counter:
    return Counter(zip(seq[:-1], seq[1:]))


class FrozenTokenizer:
    """Immutable once built. `.tokenizer_hash` identifies it; `.to_dict()` serialises it."""

    def __init__(self, merges: List[Tuple[int, int]], specials: List[str] = None):
        self.base = 256
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials or SPECIALS)
        # merged token ids start right after the byte base
        self.merge_rank: Dict[Tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.merge_new_id: Dict[Tuple[int, int], int] = {
            pair: self.base + i for i, pair in enumerate(self.merges)
        }
        n_merged = self.base + len(self.merges)
        self.special_ids: Dict[str, int] = {
            s: n_merged + i for i, s in enumerate(self.specials)
        }
        self.vocab_size = n_merged + len(self.specials)
        self.pad_id = self.special_ids["<pad>"]
        self.bos_id = self.special_ids["<bos>"]
        self.eos_id = self.special_ids["<eos>"]

    # ---- construction -------------------------------------------------------
    @classmethod
    def train(cls, texts: List[str], num_merges: int) -> "FrozenTokenizer":
        # each pretoken becomes a mutable list of ids (starting as raw bytes)
        words = [list(p) for t in texts for p in _pretokenize(t)]
        merges: List[Tuple[int, int]] = []
        next_id = 256
        for _ in range(num_merges):
            pairs: Counter = Counter()
            for w in words:
                if len(w) < 2:
                    continue
                pairs.update(zip(w[:-1], w[1:]))
            if not pairs:
                break
            # deterministic: max frequency, tie-break by smallest pair tuple
            best = min(pairs, key=lambda p: (-pairs[p], p))
            if pairs[best] < 2:
                break
            merges.append(best)
            new_id = next_id
            next_id += 1
            a, b = best
            for w in words:
                i = 0
                while i < len(w) - 1:
                    if w[i] == a and w[i + 1] == b:
                        w[i:i + 2] = [new_id]
                    else:
                        i += 1
        return cls(merges)

    # ---- (de)serialise ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"format": "tdes-bpe-v1", "base": self.base,
                "merges": [list(m) for m in self.merges], "specials": self.specials}

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenTokenizer":
        assert d["format"] == "tdes-bpe-v1", "unknown tokenizer format"
        return cls([tuple(m) for m in d["merges"]], d["specials"])

    @property
    def tokenizer_hash(self) -> str:
        return sha256_json(self.to_dict())

    # ---- encode -------------------------------------------------------------
    def _encode_word(self, ids: List[int]) -> List[int]:
        while len(ids) >= 2:
            pairs = set(zip(ids[:-1], ids[1:]))
            cand = [p for p in pairs if p in self.merge_rank]
            if not cand:
                break
            pair = min(cand, key=lambda p: self.merge_rank[p])
            new = self.merge_new_id[pair]
            a, b = pair
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for piece in _pretokenize(text):
            out.extend(self._encode_word(list(piece)))
        return out

    def decode(self, ids: List[int]) -> str:
        # invert merges to bytes; specials render as their literal name
        buf = bytearray()
        rev = {v: k for k, v in self.merge_new_id.items()}
        id_to_special = {v: k for k, v in self.special_ids.items()}

        def emit(tid: int):
            if tid in id_to_special:
                buf.extend(id_to_special[tid].encode("utf-8"))
            elif tid < self.base:
                buf.append(tid)
            else:
                a, b = rev[tid]
                emit(a)
                emit(b)

        for tid in ids:
            emit(tid)
        return buf.decode("utf-8", errors="replace")
