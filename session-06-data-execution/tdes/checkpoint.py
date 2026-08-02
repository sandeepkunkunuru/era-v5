"""Checkpoints tied to ledger offsets (`[00:39:27]`–`[00:40:27]`). A checkpoint
stores everything needed to resume at the EXACT point: model weights, RNG state,
the sampler state (stream position), and the consumption/learning ledger offsets.

"What is the meaning of exact? That is a ledger." (`[01:44:24]`)
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from .hashing import canonical, sha256_bytes, sha256_json
from .model import BigramLM
from .stream import SamplerState


def save(ckpt_dir: pathlib.Path, step: int, model: BigramLM, rng: np.random.Generator,
         sampler: SamplerState, consumption_offset: int, learning_offset: int,
         tokenizer_hash: str, master_seed: int) -> dict:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weight_bytes = model.state_bytes()
    weight_hash = sha256_bytes(weight_bytes)
    (ckpt_dir / f"ckpt_{step:06d}.model.bin").write_bytes(weight_bytes)
    meta = {
        "step": step,
        "master_seed": master_seed,
        "tokenizer_hash": tokenizer_hash,
        "weight_hash": weight_hash,
        "rng_state": rng.bit_generator.state,
        "sampler_state": sampler.to_dict(),
        "consumption_offset": consumption_offset,
        "learning_offset": learning_offset,
    }
    meta["checkpoint_id"] = sha256_json(meta)
    (ckpt_dir / f"ckpt_{step:06d}.json").write_bytes(canonical(meta))
    return meta


def load(ckpt_dir: pathlib.Path, step: int, model: BigramLM,
         rng: Optional[np.random.Generator] = None) -> dict:
    import json
    meta = json.loads((ckpt_dir / f"ckpt_{step:06d}.json").read_bytes())
    weight_bytes = (ckpt_dir / f"ckpt_{step:06d}.model.bin").read_bytes()
    # integrity: weights must match the recorded hash
    assert sha256_bytes(weight_bytes) == meta["weight_hash"], "checkpoint weight hash mismatch"
    model.load_bytes(weight_bytes)
    if rng is not None:
        rng.bit_generator.state = meta["rng_state"]
    return meta


def sampler_from(meta: dict) -> SamplerState:
    return SamplerState.from_dict(meta["sampler_state"])
