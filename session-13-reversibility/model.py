"""A 21M-parameter byte-level GPT with four interchangeable stacks.

The backbone is fixed; only the rule that carries the hidden state from one layer to the next
changes. Three of the four rules are reversible, which means the backward pass can rebuild every
layer's input from its output and the forward pass can therefore throw the activations away.

    baseline     p[l+1] = p[l] + f(p[l])                      ordinary residual; stores everything
    midpoint     p[l+1] = a*p[l-1] + 2h*f(p[l])               paper eq. 2.4 (a = 1)
    leapfrog     p[l+1] = 2p[l] - p[l-1] + h^2*f(p[l])        paper eq. 2.6, the wave form
    hamiltonian  q[l]   = a*q[l-1] + Attn(LN1(p[l-1]))        paper eq. 2.8-2.9, symplectic Euler
                 p[l]   = b*p[l-1] + MLP(LN2(q[l]))

f is the whole transformer block, as the paper defines it (eq. 2.5):

    f(p) = Attn(LN1(p)) + MLP(LN2(p + Attn(LN1(p))))

Reference: Gal, Eliasof, Turek, Ascher, Treister & Haber, "Reversing Large Language Models for
Efficient Training and Fine-Tuning", arXiv:2512.02056 (Nov 2025). Session 13 teaches the midpoint
rule and calls the alternative "Euler"; the paper's own alternatives are the leapfrog (wave) form
and the Hamiltonian (symplectic-Euler) form. Plain forward Euler is not reversible, because
recovering p[l] from p[l+1] would need f evaluated at the state being recovered.

The damping coefficient `a` is here because V4's production integrator ran a = 0.5 while the
paper's stability analysis (§3) requires |a| = 1 for a method to be stable both forwards and
backwards. train.py measures what a = 0.5 does to the reconstruction.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256          # raw bytes: the tokenizer question is Session 2's, not this one
    n_layer: int = 12
    n_head: int = 6
    d_model: int = 384
    block_size: int = 512
    stack: str = "baseline"        # baseline | midpoint | leapfrog | hamiltonian
    h: float = 0.25                # step size; V4 production used 0.25
    a: float = 1.0                 # damping; paper requires |a| = 1, V4 ran 0.5
    dropout: float = 0.0           # must be 0 for a reversible stack -- see train.py


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
    """SwiGLU, with the inner width chosen so three matrices cost what two 4x matrices would."""

    def __init__(self, cfg):
        super().__init__()
        inner = int(8 / 3 * cfg.d_model / 64 + 0.5) * 64
        self.gate = nn.Linear(cfg.d_model, inner, bias=False)
        self.up = nn.Linear(cfg.d_model, inner, bias=False)
        self.down = nn.Linear(inner, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """f(p) from the paper's eq. 2.5 -- the whole block, not half of one."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, p):
        h = self.attn(self.ln1(p))
        return h + self.mlp(self.ln2(p + h))

    # The Hamiltonian stack needs the two halves separately (eq. 2.8 and 2.9).
    def attn_part(self, p):
        return self.attn(self.ln1(p))

    def mlp_part(self, q):
        return self.mlp(self.ln2(q))


# ===================================================================== reversible stacks
#
# Each stack is one autograd Function. Forward runs under no_grad and keeps only the boundary
# states, so activation memory does not grow with depth. Backward walks the stack in reverse,
# rebuilding each layer's input from its output and re-running the block with grad enabled to get
# the parameter gradients.


def _vjp(block, p, v):
    """Gradients of <v, f(p)> wrt p and wrt block's parameters, recomputing f(p) with grad on."""
    with torch.enable_grad():
        p = p.detach().requires_grad_(True)
        out = block(p)
        params = [q for q in block.parameters()]
        grads = torch.autograd.grad(out, [p] + params, grad_outputs=v, allow_unused=False)
    return grads[0], grads[1:]


class _Midpoint(torch.autograd.Function):
    """p[l+1] = a*p[l-1] + 2h*f(p[l]); inverse p[l-1] = (p[l+1] - 2h*f(p[l])) / a."""

    @staticmethod
    def forward(ctx, p0, blocks, h, a, *params):
        with torch.no_grad():
            prev = p0
            cur = p0 + h * blocks[0](p0)              # Euler bootstrap, as V4's config did
            for blk in blocks[1:]:
                prev, cur = cur, a * prev + 2 * h * blk(cur)
        ctx.blocks, ctx.h, ctx.a = blocks, h, a
        ctx.save_for_backward(prev, cur)
        return cur

    @staticmethod
    def backward(ctx, g_out):
        blocks, h, a = ctx.blocks, ctx.h, ctx.a
        prev, cur = ctx.saved_tensors                 # prev = p[L-2], cur = p[L-1]
        pgrads = [torch.zeros_like(q) for blk in blocks for q in blk.parameters()]
        sizes = [len(list(blk.parameters())) for blk in blocks]
        g_cur, g_next = torch.zeros_like(g_out), g_out
        off = sum(sizes)
        for i in range(len(blocks) - 1, 0, -1):
            blk = blocks[i]
            back = prev                               # p[l], the state f was evaluated at
            fwd = cur                                 # p[l+1]
            recovered = (fwd - 2 * h * blk(back)) / a  # p[l-1]
            dp, dps = _vjp(blk, back, 2 * h * g_next)
            off -= sizes[i]
            for j, gp in enumerate(dps):
                pgrads[off + j] += gp
            g_cur, g_next = g_cur + dp, a * g_next
            g_cur, g_next = g_next, g_cur             # shift one layer down
            cur, prev = back, recovered
        # the bootstrap step: p[1] = p[0] + h*f0(p[0])
        blk = blocks[0]
        dp, dps = _vjp(blk, prev, h * g_next)
        for j, gp in enumerate(dps):
            pgrads[j] += gp
        g_p0 = g_cur + g_next + dp
        return (g_p0, None, None, None, *pgrads)


class _Leapfrog(torch.autograd.Function):
    """p[l+1] = 2p[l] - p[l-1] + h^2*f(p[l]); inverse p[l-1] = 2p[l] - p[l+1] + h^2*f(p[l])."""

    @staticmethod
    def forward(ctx, p0, blocks, h, a, *params):
        with torch.no_grad():
            prev = p0
            cur = p0 + 0.5 * h * h * blocks[0](p0)    # one half-step to start the recurrence
            for blk in blocks[1:]:
                prev, cur = cur, 2 * cur - prev + h * h * blk(cur)
        ctx.blocks, ctx.h = blocks, h
        ctx.save_for_backward(prev, cur)
        return cur

    @staticmethod
    def backward(ctx, g_out):
        blocks, h = ctx.blocks, ctx.h
        prev, cur = ctx.saved_tensors
        pgrads = [torch.zeros_like(q) for blk in blocks for q in blk.parameters()]
        sizes = [len(list(blk.parameters())) for blk in blocks]
        g_cur, g_next = torch.zeros_like(g_out), g_out
        off = sum(sizes)
        for i in range(len(blocks) - 1, 0, -1):
            blk = blocks[i]
            back, fwd = prev, cur
            recovered = 2 * back - fwd + h * h * blk(back)
            dp, dps = _vjp(blk, back, h * h * g_next)
            off -= sizes[i]
            for j, gp in enumerate(dps):
                pgrads[off + j] += gp
            new_g_cur = -g_next                        # dL/dp[l-1]
            new_g_next = g_cur + 2 * g_next + dp       # dL/dp[l]
            g_cur, g_next = new_g_cur, new_g_next
            cur, prev = back, recovered
        blk = blocks[0]
        dp, dps = _vjp(blk, prev, 0.5 * h * h * g_next)
        for j, gp in enumerate(dps):
            pgrads[j] += gp
        g_p0 = g_cur + g_next + dp
        return (g_p0, None, None, None, *pgrads)


class _Hamiltonian(torch.autograd.Function):
    """q[l] = a*q[l-1] + Attn(LN1(p[l-1])); p[l] = b*p[l-1] + MLP(LN2(q[l])). Inverted in reverse
    order: p[l-1] = (p[l] - MLP(LN2(q[l])))/b first, then q[l-1] = (q[l] - Attn(LN1(p[l-1])))/a."""

    @staticmethod
    def forward(ctx, p0, blocks, h, a, *params):
        with torch.no_grad():
            p, q = p0, torch.zeros_like(p0)
            for blk in blocks:
                q = a * q + blk.attn_part(p)
                p = a * p + blk.mlp_part(q)
        ctx.blocks, ctx.a = blocks, a
        ctx.save_for_backward(p, q)
        return p

    @staticmethod
    def backward(ctx, g_out):
        blocks, a = ctx.blocks, ctx.a
        p, q = ctx.saved_tensors
        pgrads = [torch.zeros_like(x) for blk in blocks for x in blk.parameters()]
        sizes = [len(list(blk.parameters())) for blk in blocks]
        gp, gq = g_out, torch.zeros_like(g_out)
        off = sum(sizes)
        for i in range(len(blocks) - 1, -1, -1):
            blk = blocks[i]
            off -= sizes[i]
            # undo eq. 2.9
            with torch.enable_grad():
                qd = q.detach().requires_grad_(True)
                out = blk.mlp_part(qd)
                params = [x for x in blk.parameters()]
                grads = torch.autograd.grad(out, [qd] + params, grad_outputs=gp, allow_unused=True)
            p_prev = (p - out.detach()) / a
            gq = gq + grads[0]
            for j, gpar in enumerate(grads[1:]):
                if gpar is not None:
                    pgrads[off + j] += gpar
            # undo eq. 2.8
            with torch.enable_grad():
                pd = p_prev.detach().requires_grad_(True)
                out2 = blk.attn_part(pd)
                grads2 = torch.autograd.grad(out2, [pd] + params, grad_outputs=gq, allow_unused=True)
            q_prev = (q - out2.detach()) / a
            gp = a * gp + grads2[0]
            gq = a * gq
            for j, gpar in enumerate(grads2[1:]):
                if gpar is not None:
                    pgrads[off + j] += gpar
            p, q = p_prev, q_prev
        return (gp, None, None, None, *pgrads)


STACKS = {"midpoint": _Midpoint, "leapfrog": _Leapfrog, "hamiltonian": _Hamiltonian}


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        p = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        if self.cfg.stack == "baseline":
            for blk in self.blocks:
                p = p + blk(p)
        else:
            fn = STACKS[self.cfg.stack]
            params = [q for blk in self.blocks for q in blk.parameters()]
            p = fn.apply(p, self.blocks, self.cfg.h, self.cfg.a, *params)
        logits = self.head(self.ln_f(p))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def reconstruction_error(self, idx):
        """How far the rebuilt states drift from the true ones -- the thing that must hold for a
        reversible stack to be correct at all. Returns the largest relative error over layers."""
        p = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        cfg = self.cfg
        true = []
        if cfg.stack == "midpoint":
            prev, cur = p, p + cfg.h * self.blocks[0](p)
            true.append(prev)
            for blk in self.blocks[1:]:
                true.append(cur)
                prev, cur = cur, cfg.a * prev + 2 * cfg.h * blk(cur)
            worst = 0.0
            for i in range(len(self.blocks) - 1, 0, -1):
                rebuilt = (cur - 2 * cfg.h * self.blocks[i](prev)) / cfg.a
                ref = true[i - 1]
                worst = max(worst, ((rebuilt - ref).norm() / ref.norm().clamp_min(1e-12)).item())
                cur, prev = prev, rebuilt
            return worst
        if cfg.stack == "leapfrog":
            prev, cur = p, p + 0.5 * cfg.h ** 2 * self.blocks[0](p)
            true.append(prev)
            for blk in self.blocks[1:]:
                true.append(cur)
                prev, cur = cur, 2 * cur - prev + cfg.h ** 2 * blk(cur)
            worst = 0.0
            for i in range(len(self.blocks) - 1, 0, -1):
                rebuilt = 2 * prev - cur + cfg.h ** 2 * self.blocks[i](prev)
                ref = true[i - 1]
                worst = max(worst, ((rebuilt - ref).norm() / ref.norm().clamp_min(1e-12)).item())
                cur, prev = prev, rebuilt
            return worst
        if cfg.stack == "hamiltonian":
            q = torch.zeros_like(p)
            states = [(p, q)]
            for blk in self.blocks:
                q = cfg.a * q + blk.attn_part(p)
                p = cfg.a * p + blk.mlp_part(q)
                states.append((p, q))
            worst = 0.0
            for i in range(len(self.blocks) - 1, -1, -1):
                blk = self.blocks[i]
                p_prev = (p - blk.mlp_part(q)) / cfg.a
                q_prev = (q - blk.attn_part(p_prev)) / cfg.a
                ref_p, ref_q = states[i]
                worst = max(worst, ((p_prev - ref_p).norm() / ref_p.norm().clamp_min(1e-12)).item())
                p, q = p_prev, q_prev
            return worst
        return 0.0
