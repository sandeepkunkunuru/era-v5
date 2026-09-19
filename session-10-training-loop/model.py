"""A small GPT, deliberately boring.

The assignment is about what happens *after* the hidden state, so the backbone here is
the plainest pre-norm transformer that will train: RMSNorm, SwiGLU, causal self-attention,
learned positions. Session 9 §2 is the reference for every choice.

The only switch that matters is `tie_weights`, because requirement 6 asks us to price it.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    block_size: int = 256
    tie_weights: bool = True
    n_extra_heads: int = 0      # Part 2: heads predicting t+2, t+3, ...


class RMSNorm(nn.Module):
    """x / rms(x) * g -- LayerNorm without the centring (Session 9 §2)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Compute the norm in *at least* fp32. The Session 9 version used x.float(), which is an
        # upcast for bf16 but a silent DOWNCAST for an fp64 model -- the Session 10 gradient check
        # (requirement 2) caught it: fp64 finite differences agreed with backward() to 1 figure.
        up = x.to(torch.promote_types(x.dtype, torch.float32))
        rms = up.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (up * rms).type_as(x) * self.g


class SwiGLU(nn.Module):
    """down(silu(gate(h)) * up(h)) -- three matrices, so d_ff shrinks to pay for the third."""

    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.d_head = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # W_O is a full [D, D] matrix, and it is what mixes the heads back together --
        # the heads do not each own a slice of the residual stream.
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, D))


class Block(nn.Module):
    """Pre-norm: the norm sits on the branch, never on the residual stream."""

    def __init__(self, cfg):
        super().__init__()
        d_ff = int(2 * (4 * cfg.d_model) / 3 / 64) * 64   # ~8/3 D, rounded to a multiple of 64
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg.d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight
        # Part 2: each extra head predicts one position further out.
        self.extra_heads = nn.ModuleList(
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(cfg.n_extra_heads)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def hidden(self, idx):
        """tokens [B,T] -> hidden state [B,T,D]. Everything upstream of the output head."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.norm_f(x)

    def forward(self, idx):
        return self.lm_head(self.hidden(idx))

    def n_params(self, trainable_only=True):
        seen, total = set(), 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:      # a tied weight is one matrix, counted once
                continue
            seen.add(id(p))
            total += p.numel()
        return total
