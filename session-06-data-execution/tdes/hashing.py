"""Stable hashing + canonical JSON. Every content hash in the system flows through
here, so "same data + same code => same hash on any machine, any day" holds — the
determinism the ledger depends on (Session 6 `[00:24:18]`, `[02:23:06]`)."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace jitter, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical(obj))


def sha256_ints(ids) -> str:
    """Hash a token-id array (list/np.ndarray) via its raw uint32 little-endian bytes."""
    import numpy as np
    a = np.asarray(ids, dtype="<u4")
    return sha256_bytes(a.tobytes())


def short(h: str, n: int = 12) -> str:
    return h[:n]


def derive_seed(*parts: Any) -> int:
    """A 63-bit seed derived deterministically from any parts (ints/strs/tuples).
    Used to seed per-step / per-lane RNGs as a pure function of position, so the
    stream is reconstructable from state alone."""
    h = hashlib.sha256(canonical(list(parts))).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)
