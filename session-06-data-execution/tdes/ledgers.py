"""Append-only, hash-chained ledgers — the "accounting book" the whole run rests on
(`[00:40:27]`, `[02:05:46]`). Two instances are used:

  * consumption ledger — what was SENT (batch id, hash, shards, lanes, masks, policy);
  * learning ledger    — what came BACK (loss per sequence + token-level samples).

Each entry chains its predecessor's hash, so the ledger is tamper-evident and its
integrity is auditable. `offset` ties a checkpoint to an exact ledger position;
`truncate_to` drops post-checkpoint entries on resume so nothing is duplicated.
"""
from __future__ import annotations

import pathlib
from typing import List

from .hashing import canonical, sha256_bytes


class Ledger:
    GENESIS = "0" * 64

    def __init__(self, name: str):
        self.name = name
        self.entries: List[dict] = []

    def head_hash(self) -> str:
        return self.entries[-1]["entry_hash"] if self.entries else self.GENESIS

    def append(self, record: dict) -> dict:
        prev = self.head_hash()
        body = canonical({"record": record, "prev_hash": prev})
        entry = {**record, "prev_hash": prev, "entry_hash": sha256_bytes(body)}
        self.entries.append(entry)
        return entry

    def offset(self) -> int:
        return len(self.entries)

    def truncate_to(self, offset: int) -> int:
        """Drop entries at index >= offset (resume after a checkpoint). Returns #dropped."""
        dropped = len(self.entries) - offset
        if dropped > 0:
            self.entries = self.entries[:offset]
        return max(0, dropped)

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for e in self.entries:
            record = {k: v for k, v in e.items() if k not in ("prev_hash", "entry_hash")}
            body = canonical({"record": record, "prev_hash": prev})
            if e["prev_hash"] != prev or e["entry_hash"] != sha256_bytes(body):
                return False
            prev = e["entry_hash"]
        return True

    def write(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for e in self.entries:
                f.write(canonical(e) + b"\n")

    def find_step(self, step: int) -> dict:
        for e in self.entries:
            if e.get("step") == step:
                return e
        raise KeyError(f"step {step} not in ledger {self.name}")
