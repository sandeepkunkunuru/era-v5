"""A dense byte-level GPT and the mixture-of-experts model it is grown into.

The dense model is the one Session 13 used: 12 layers, d_model 384, 6 heads, block 512, byte
vocabulary, SwiGLU feed-forward with an inner width of 1,024. The inner width matters here: 1,024
cuts into exactly 8 slices of 128, which is what makes the growth step arithmetic rather than
guesswork.

    dense       one feed-forward network per layer, inner width 1024, every weight used
    moe         32 experts per layer, inner width 128, top-8 -> active inner width 1024

So the mixture-of-experts model uses the same width per token as the dense model and stores four
times as many feed-forward parameters. That is the trade the session is about: capacity follows the
total, compute follows the active.

Growing (Session 14 S15, and the V4 cookbook's own recipe):

    random      experts initialised from scratch; the control, nothing is inherited
    clone       8 distinct slices, each copied 4 times -> 8 clone families of 4 members
    partition   the 1,024 neurons are permuted per tile and cut into 8 experts of 128
    drop        partition with half of each expert's neurons redrawn (drop-upcycling, r = 0.5)

`scale_down` multiplies the inherited down-projection by top_k. A fresh router gives every expert
about the same score, so the 8 kept weights are about 1/8 each and the layer's output is about one
eighth of the dense layer's. Scaling the down-projection by 8 cancels that. This is the same
correction the brief calls a routed scaling factor (2.5 in DeepSeek-V3, 1.5 in V4.1-Flash), except
that here the right factor is known in closed form, and train.py's exactness check confirms it
reproduces the dense layer to float precision.

Balancing:

    none        hard top-k on the router's own scores
    aux         Switch Transformer's loss, alpha * N * sum_i f_i * P_i   (arXiv:2101.03961)
    bias        auxiliary-loss-free: b_i += gamma * sign(mean_load - load_i), used only for
                choosing, never for weighting                            (arXiv:2408.15664)

The router runs in fp32 whatever the rest of the model runs in, because Switch Transformer's base
model diverged with a 16-bit router.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256          # raw bytes; the tokenizer question is Session 2's
    n_layer: int = 12
    n_head: int = 6
    d_model: int = 384
    block_size: int = 512
    # mixture-of-experts; ignored when moe is False
    moe: bool = False
    n_expert: int = 32
    top_k: int = 8
    expert_inner: int = 128        # 8 * 128 = 1024 = the dense inner width
    balance: str = "bias"          # none | aux | bias
    aux_alpha: float = 0.01        # Switch's chosen value
    bias_gamma: float = 0.001      # DeepSeek-V3's value
    prob_routing_steps: int = 0    # steps of probabilistic selection before hard top-k
    scope: str = "batch"           # batch | micro -- the window the load is counted over


def dense_inner(d_model):
    """Session 13's rule: three matrices costing what two 4x matrices would. 384 -> 1024."""
    return int(8 / 3 * d_model / 64 + 0.5) * 64


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        up = x.to(torch.promote_types(x.dtype, torch.float32))
        return (up * torch.rsqrt(up.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.g


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.d_head = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    """The dense feed-forward network. One of these per layer is what MoE replaces."""

    def __init__(self, cfg):
        super().__init__()
        inner = dense_inner(cfg.d_model)
        self.inner = inner
        self.gate = nn.Linear(cfg.d_model, inner, bias=False)
        self.up = nn.Linear(cfg.d_model, inner, bias=False)
        self.down = nn.Linear(inner, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================================ the MoE layer


class MoE(nn.Module):
    """A router and n_expert small SwiGLU networks; each token runs top_k of them.

    Experts are stored as three stacked tensors rather than a list of Linears, so that growing the
    model is a tensor operation and the dispatch loop can index them directly. The loop over experts
    is the honest thing to do: it computes only the experts a token chose, which is the whole point.
    A dense einsum over all 32 experts would be faster on this card and would compute 4x the work.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        E, I, D = cfg.n_expert, cfg.expert_inner, cfg.d_model
        self.E, self.I, self.k = E, I, cfg.top_k
        self.router = nn.Linear(D, E, bias=False)
        # Switch Transformer initialised the router at one tenth of the usual scale, and said that
        # at the usual scale the quality varied widely between runs. A zero router would make top-k
        # fall back on how the kernel breaks ties, which is not a routing decision at all.
        nn.init.normal_(self.router.weight, std=0.002)
        self.w_gate = nn.Parameter(torch.empty(E, D, I))
        self.w_up = nn.Parameter(torch.empty(E, D, I))
        self.w_down = nn.Parameter(torch.empty(E, I, D))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)
        # Balancing state. `bias` is not a parameter: no gradient ever reaches it.
        self.register_buffer("bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))        # the bias update's window
        self.register_buffer("mon", torch.zeros(E))         # a monitor window, cleared on request
        self.last_aux = None

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        N = flat.shape[0]

        logits = self.router(flat.float())                  # fp32 router
        scores = logits.softmax(-1)

        choose = scores + self.bias if self.cfg.balance == "bias" else scores
        if self.training and self.cfg.prob_routing_steps > 0:
            # Probabilistic selection: sample k distinct experts with probability proportional to
            # the score, so near-identical clones all receive some gradient before the router
            # settles. This is the fix the V4 cookbook reports for collapse along clone families.
            idx = torch.multinomial(scores.clamp_min(1e-9), self.k, replacement=False)
        else:
            idx = choose.topk(self.k, dim=-1).indices

        w = scores.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)                     # weights add up to one

        out = torch.zeros_like(flat)
        flat_idx = idx.reshape(-1)
        tok = torch.arange(N, device=x.device).repeat_interleave(self.k)
        wt = w.reshape(-1).to(flat.dtype)
        counts = torch.bincount(flat_idx, minlength=self.E)
        order = torch.argsort(flat_idx)
        bounds = torch.cumsum(counts, 0).tolist()
        start = 0
        for e in range(self.E):
            end = bounds[e]
            if end == start:
                continue
            sel = order[start:end]
            start = end
            t = tok[sel]
            xi = flat[t]
            h = F.silu(xi @ self.w_gate[e]) * (xi @ self.w_up[e])
            y = (h @ self.w_down[e]) * wt[sel].unsqueeze(1)
            out.index_add_(0, t, y.to(out.dtype))

        if self.training:
            self.load += counts.detach().float()
            self.mon += counts.detach().float()
            if self.cfg.balance == "aux":
                f = counts.detach().float() / (N * self.k)
                P = scores.mean(0)
                self.last_aux = self.cfg.aux_alpha * self.E * (f * P).sum()
        return out.view(B, T, D)

    # -------------------------------------------------------------- balancing bookkeeping
    def step_bias(self, gamma):
        """b_i += gamma * sign(mean_load - load_i), then clear the window."""
        if self.load.sum() > 0:
            mean = self.load.mean()
            self.bias += gamma * torch.sign(mean - self.load)
        self.load.zero_()

    def load_stats(self):
        """MaxVio and the number of dead experts over the monitor window."""
        tot = self.mon.sum()
        if tot == 0:
            return dict(maxvio=float("nan"), dead=self.E, share=[])
        share = (self.mon / tot).tolist()
        mean = 1.0 / self.E
        return dict(maxvio=(max(share) - mean) / mean, dead=int((self.mon == 0).sum()), share=share)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = MoE(cfg) if cfg.moe else MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight             # tied
        self.apply(self._init)
        for blk in self.blocks:                        # apply() would overwrite the router scale
            if isinstance(blk.mlp, MoE):
                nn.init.normal_(blk.mlp.router.weight, std=0.002)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # --------------------------------------------------------------------- MoE utilities
    def moe_layers(self):
        return [blk.mlp for blk in self.blocks if isinstance(blk.mlp, MoE)]

    def aux_loss(self):
        parts = [m.last_aux for m in self.moe_layers() if m.last_aux is not None]
        return sum(parts) if parts else None

    def step_bias(self, gamma):
        for m in self.moe_layers():
            m.step_bias(gamma)

    def reset_mon(self):
        for m in self.moe_layers():
            m.mon.zero_()

    def load_stats(self):
        """Mean MaxVio and total dead experts across layers."""
        st = [m.load_stats() for m in self.moe_layers()]
        if not st:
            return {}
        return dict(maxvio=sum(s["maxvio"] for s in st) / len(st),
                    maxvio_worst=max(s["maxvio"] for s in st),
                    dead=sum(s["dead"] for s in st),
                    dead_total=sum(m.E for m in self.moe_layers()),
                    per_layer_share=[s["share"] for s in st])

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        if not self.cfg.moe:
            return dict(total=total, active=total)
        per_expert = 3 * self.cfg.d_model * self.cfg.expert_inner
        idle = (self.cfg.n_expert - self.cfg.top_k) * per_expert * self.cfg.n_layer
        return dict(total=total, active=total - idle)


# ============================================================================ growing


def upcycle(dense, method, cfg, scale_down=False, seed=0):
    """Build an MoE model of shape `cfg` out of a trained dense model.

    Returns the new model. Attention, embeddings and norms are copied verbatim in every method;
    only the feed-forward weights differ, which is what keeps the comparison clean.
    """
    assert cfg.moe
    g = torch.Generator(device="cpu").manual_seed(seed)
    moe = GPT(cfg)
    dsd, msd = dense.state_dict(), moe.state_dict()
    for name, t in dsd.items():
        if ".mlp." not in name and name in msd:
            msd[name].copy_(t)

    E, I, D = cfg.n_expert, cfg.expert_inner, cfg.d_model
    inner = dense_inner(D)
    slices = inner // I                       # 8
    assert E % slices == 0, f"{E} experts does not tile {slices} slices"
    tiles = E // slices                       # 4
    std = 0.02

    for li, blk in enumerate(dense.blocks):
        mlp, dst = blk.mlp, moe.blocks[li].mlp
        # Dense weights as (D, inner) / (inner, D) so a slice is a contiguous block of neurons.
        Wg, Wu = mlp.gate.weight.t().contiguous(), mlp.up.weight.t().contiguous()
        Wd = mlp.down.weight.t().contiguous()
        for t in range(tiles):
            if method == "random":
                continue
            if method == "clone":
                perm = torch.arange(inner)                       # every tile the same 8 slices
            else:
                perm = torch.arange(inner) if t == 0 else torch.randperm(inner, generator=g)
            for s in range(slices):
                e = t * slices + s
                cols = perm[s * I:(s + 1) * I]
                dst.w_gate.data[e] = Wg[:, cols]
                dst.w_up.data[e] = Wu[:, cols]
                dst.w_down.data[e] = Wd[cols, :]
                if method == "drop":
                    # Drop-upcycling: redraw a fraction r = 0.5 of this expert's neurons.
                    keep = torch.randperm(I, generator=g)[: I // 2]
                    mask = torch.ones(I, dtype=torch.bool)
                    mask[keep] = False                            # True = redraw
                    n = int(mask.sum())
                    dst.w_gate.data[e][:, mask] = torch.randn(D, n, generator=g) * std
                    dst.w_up.data[e][:, mask] = torch.randn(D, n, generator=g) * std
                    dst.w_down.data[e][mask, :] = torch.randn(n, D, generator=g) * std
        if scale_down and method != "random":
            dst.w_down.data *= cfg.top_k
    return moe
