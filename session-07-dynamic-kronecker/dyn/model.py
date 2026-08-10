"""A small decoder-only transformer whose *input path* is the only thing that varies.

Three input paths are available and they are drop-in interchangeable — each takes
[B,T] integer ids and returns [B,T,d_model] floats, exactly the contract Session 7
describes ("everything about how the row was manufactured is private to it").

  DenseEmbedding      — the control arm: a real V x d table, one trainable row per token.
  KroneckerEmbedding  — a frozen byte code + one shared trainable projection.

Every arm uses an untied dense output head (V5's decision), so the *only* difference
between runs is how a token id becomes a vector.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codecs import ByteCodec


class DenseEmbedding(nn.Module):
    """Control arm: `nn.Embedding`. Trainable parameters scale with vocabulary."""

    def __init__(self, vocab: list[str], d_model: int):
        super().__init__()
        self.table = nn.Embedding(len(vocab), d_model)
        nn.init.normal_(self.table.weight, std=0.02)

    @property
    def input_path_params(self) -> int:
        return self.table.weight.numel()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table(ids)


class KroneckerEmbedding(nn.Module):
    """A frozen byte code plus one shared trainable projection.

    The code table is a *buffer*, never a parameter — it holds no trainable weights and
    is identical on every run. Trainable parameter count is `code_dim * d_model` and does
    not depend on the vocabulary size at all.
    """

    def __init__(self, vocab: list[str], d_model: int, codec: ByteCodec):
        super().__init__()
        self.codec = codec
        self.register_buffer("code", codec.encode(vocab), persistent=False)
        self.proj = nn.Linear(codec.code_dim, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    @property
    def input_path_params(self) -> int:
        return self.proj.weight.numel()          # the buffer contributes zero

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.code[ids])


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, embedder: nn.Module, vocab_size: int, d_model: int = 256,
                 n_layer: int = 4, n_head: int = 4, block_size: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.embed = embedder
        self.block_size = block_size
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)   # untied, per V5
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    @property
    def input_path_params(self) -> int:
        return self.embed.input_path_params

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = ids.shape
        x = self.embed(ids) * math.sqrt(1.0)
        x = x + self.pos(torch.arange(T, device=ids.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss


def build(kind: str, vocab: list[str], d_model: int, codec: ByteCodec | None = None,
          **kw) -> TinyLM:
    embedder = (DenseEmbedding(vocab, d_model) if kind == "dense"
                else KroneckerEmbedding(vocab, d_model, codec))
    return TinyLM(embedder, len(vocab), d_model=d_model, **kw)
