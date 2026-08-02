"""A tiny, deterministic model — "training is just fake training" (`[02:23:06]`),
so the model is deliberately minimal. It is a real bigram LM: logits for the next
token depend on the current token via a (V×V) table trained with SGD.

Two properties matter for the ledgers:
  * it produces REAL per-token cross-entropy loss, masked by the loss mask;
  * with a zero-initialised table the initial loss is exactly **ln(V)** — the
    lecture's −ln(1/vocab) = 11.78 for a 131072 vocab (`[01:58:35]`).

No randomness in the update, so training is bit-reproducible given the batch.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


class BigramLM:
    def __init__(self, vocab_size: int, lr: float = 0.5):
        self.V = vocab_size
        self.lr = lr
        self.M = np.zeros((vocab_size, vocab_size), dtype=np.float32)  # zero => loss=ln(V)

    # ---- state (for checkpoints) -------------------------------------------
    def state_bytes(self) -> bytes:
        return self.M.astype("<f4").tobytes()

    def load_bytes(self, b: bytes) -> None:
        self.M = np.frombuffer(b, dtype="<f4").reshape(self.V, self.V).copy()

    def weight_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.M.tobytes()).hexdigest()

    # ---- one training step --------------------------------------------------
    def step(self, tokens: np.ndarray, loss_mask: np.ndarray,
             train: bool = True) -> Dict:
        """tokens (B,T) uint; loss_mask (B,T) bool where position i trains to predict
        i+1. Returns mean loss + per-position loss for the learning ledger."""
        if tokens.shape[0] == 0:
            return {"mean_loss": 0.0, "n_loss_tokens": 0, "pos_loss": None}
        inp = tokens[:, :-1].astype(np.int64)      # (B,T-1)
        tgt = tokens[:, 1:].astype(np.int64)
        m = loss_mask[:, :-1]                       # align: pos i predicts i+1
        logits = self.M[inp]                        # (B,T-1,V)
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=-1, keepdims=True)
        # per-position loss = -log p[target]
        B, Tm1, V = p.shape
        bi, ti = np.meshgrid(np.arange(B), np.arange(Tm1), indexing="ij")
        p_tgt = p[bi, ti, tgt]
        pos_loss = -np.log(np.clip(p_tgt, 1e-12, 1.0))
        pos_loss = np.where(m, pos_loss, 0.0)
        n = int(m.sum())
        mean_loss = float(pos_loss.sum() / n) if n else 0.0

        if train and n:
            grad = p.copy()
            grad[bi, ti, tgt] -= 1.0
            grad *= m[:, :, None]                   # zero non-loss positions
            # accumulate row updates: M[inp] -= lr * grad / n
            np.add.at(self.M, inp.reshape(-1),
                      (-self.lr / n) * grad.reshape(-1, V))
        return {"mean_loss": mean_loss, "n_loss_tokens": n,
                "pos_loss": pos_loss, "loss_mask_used": m}

    @staticmethod
    def initial_loss(vocab_size: int) -> float:
        return math.log(vocab_size)
