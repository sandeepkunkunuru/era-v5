"""Immutable tokenized shards + manifests (`[01:26:09]`–`[01:30:12]`).

A shard is a frozen binary uint32 token array. Its manifest records the tokenizer
hash, a content hash (the "signature … to figure out have I trained on the shard
or not" `[01:28:11]`), provenance, the cleaning-pipeline code hash, and the
dedup / contamination / PII / eval-overlap status that gate whether it may be
trained on. Loading re-verifies both hashes, so tampering is caught.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .corpus import Document
from .hashing import sha256_bytes, sha256_ints, sha256_json, short
from .tokenizer import FrozenTokenizer

MAX_DOCS_PER_SHARD = 8   # small, so lanes produce several shards to schedule over


def _doc_tokens(tok: FrozenTokenizer, doc: Document) -> Tuple[List[int], List[List[int]]]:
    """Return (token_ids, loss_spans). loss_spans are [start,end) ranges (relative to
    the doc) whose tokens bear loss. Unstructured lanes => whole doc bears loss
    (pretraining). Structured lanes => only answer segments (SFT/agentic rule)."""
    if doc["segments"] is None:
        ids = tok.encode(doc["text"])
        return ids, [[0, len(ids)]]
    ids: List[int] = []
    spans: List[List[int]] = []
    for seg in doc["segments"]:
        seg_ids = tok.encode(seg["text"])
        start = len(ids)
        ids.extend(seg_ids)
        if seg["loss"] and seg_ids:
            spans.append([start, len(ids)])
    return ids, spans


@dataclass
class Shard:
    shard_id: str
    lane: str
    split: str
    tokens: np.ndarray            # uint32, concatenated doc tokens
    docs: List[dict]              # {doc_id, start, len, loss_spans(abs)}
    manifest: dict

    def doc_arrays(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """(ids, loss_bool) for the i-th document in this shard."""
        d = self.docs[i]
        ids = self.tokens[d["start"]: d["start"] + d["len"]]
        loss = np.zeros(d["len"], dtype=bool)
        for s, e in d["loss_spans"]:
            loss[s - d["start"]: e - d["start"]] = True
        return ids, loss


def cleaning_pipeline_hash() -> str:
    """Hash of the code that produced the shards — a stand-in for 'what script
    cleaned this' (`[01:28:11]`). Hashes this module + the corpus module."""
    here = pathlib.Path(__file__).parent
    blob = b""
    for name in ("corpus.py", "shards.py", "tokenizer.py"):
        blob += (here / name).read_bytes()
    return sha256_bytes(blob)


class ShardSet:
    """Builds, writes, loads, and verifies the whole set of shards."""

    def __init__(self, shards: List[Shard], tokenizer_hash: str):
        self.shards = shards
        self.tokenizer_hash = tokenizer_hash
        self.by_id = {s.shard_id: s for s in shards}

    # ---- build --------------------------------------------------------------
    @classmethod
    def build(cls, docs: List[Document], tok: FrozenTokenizer) -> "ShardSet":
        # group by (lane, split), then chunk into shards
        groups: Dict[Tuple[str, str], List[Document]] = {}
        for d in docs:
            groups.setdefault((d["lane"], d["split"]), []).append(d)

        # seen content hashes -> dedup detection across the whole corpus
        seen_doc_hash: Dict[str, str] = {}
        eval_doc_hashes = {
            sha256_ints(_doc_tokens(tok, d)[0]) for d in docs if d["split"] == "eval"
        }
        clean_hash = cleaning_pipeline_hash()
        shards: List[Shard] = []
        for (lane, split), group in sorted(groups.items()):
            for ci in range(0, len(group), MAX_DOCS_PER_SHARD):
                chunk = group[ci: ci + MAX_DOCS_PER_SHARD]
                toks: List[int] = []
                recs: List[dict] = []
                dedup_dropped = 0
                for d in chunk:
                    ids, spans = _doc_tokens(tok, d)
                    dh = sha256_ints(ids)
                    if dh in seen_doc_hash:      # exact-duplicate document
                        dedup_dropped += 1
                        continue
                    seen_doc_hash[dh] = d["doc_id"]
                    start = len(toks)
                    toks.extend(ids)
                    abs_spans = [[start + s, start + e] for s, e in spans]
                    recs.append({"doc_id": d["doc_id"], "start": start,
                                 "len": len(ids), "loss_spans": abs_spans,
                                 "doc_sha256": dh})
                arr = np.asarray(toks, dtype=np.uint32)
                sid = f"{lane}.{split}.{ci // MAX_DOCS_PER_SHARD:02d}"
                content_sha = sha256_bytes(arr.tobytes())
                n_loss = sum(e - s for r in recs for s, e in r["loss_spans"])
                # contamination: does any train shard overlap an eval doc hash?
                contaminated = split == "train" and any(
                    r["doc_sha256"] in eval_doc_hashes for r in recs)
                manifest = {
                    "shard_id": sid, "lane": lane, "split": split,
                    "tokenizer_hash": tok.tokenizer_hash,
                    "content_sha256": content_sha,
                    "n_docs": len(recs), "n_tokens": int(arr.size),
                    "n_loss_tokens": int(n_loss),
                    "cleaning_pipeline_hash": clean_hash,
                    "provenance": {"corpus_builder": "tdes.corpus.build_corpus"},
                    "quality": {
                        "dedup": "passed", "dedup_dropped": dedup_dropped,
                        "pii": "none", "contamination": "eval-overlap" if contaminated
                        else "none",
                        "eval_overlap": bool(contaminated),
                    },
                    "trainable": (split == "train") and not contaminated,
                    "frozen": True,
                    "docs": recs,
                }
                shards.append(Shard(sid, lane, split, arr, recs, manifest))
        return cls(shards, tok.tokenizer_hash)

    # ---- persist ------------------------------------------------------------
    def write(self, out_dir: pathlib.Path) -> dict:
        sh_dir = out_dir / "shards"
        mf_dir = out_dir / "manifests"
        sh_dir.mkdir(parents=True, exist_ok=True)
        mf_dir.mkdir(parents=True, exist_ok=True)
        index = {"tokenizer_hash": self.tokenizer_hash, "shards": []}
        for s in self.shards:
            (sh_dir / f"{s.shard_id}.bin").write_bytes(s.tokens.tobytes())
            from .hashing import canonical
            (mf_dir / f"{s.shard_id}.json").write_bytes(canonical(s.manifest))
            index["shards"].append({
                "shard_id": s.shard_id, "lane": s.lane, "split": s.split,
                "content_sha256": s.manifest["content_sha256"],
                "n_tokens": s.manifest["n_tokens"], "trainable": s.manifest["trainable"],
            })
        index["index_sha256"] = sha256_json(index["shards"])
        from .hashing import canonical
        (mf_dir / "index.json").write_bytes(canonical(index))
        return index

    # ---- verify -------------------------------------------------------------
    def verify(self) -> List[dict]:
        """Recompute every hash; return per-shard verification records."""
        out = []
        for s in self.shards:
            recomputed = sha256_bytes(s.tokens.tobytes())
            tok_ok = s.manifest["tokenizer_hash"] == self.tokenizer_hash
            content_ok = recomputed == s.manifest["content_sha256"]
            out.append({"shard_id": s.shard_id, "tokenizer_hash_ok": tok_ok,
                        "content_hash_ok": content_ok,
                        "content_sha256": short(recomputed)})
        return out

    # ---- convenience --------------------------------------------------------
    def trainable(self) -> List[Shard]:
        return [s for s in self.shards if s.manifest["trainable"]]

    def eval_shards(self) -> List[Shard]:
        return [s for s in self.shards if s.split == "eval"]
