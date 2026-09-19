"""ERA V5 · Session 10 — make a small model and a real loop tell the truth about itself.

One function per requirement of the assignment:

    req1  print every tensor shape in one step, with what each dimension means
    req2  verify one gradient by hand: nudge a weight, compare against backward()
    req3  break gradient accumulation on purpose (average of averages, unequal lengths)
    req4  log the grad norm every step and find a step where it moved before the loss
    req5  compute MFU honestly, and measure what costs the distance to 40%
    req6  0.1 in fp32, bf16 and fp8 E4M3, bit by bit, by hand

    python training_loop.py          # full run: writes run.log, results.json, figures/
    python training_loop.py --quick  # same structure, fewer steps, no torch.compile
"""
import argparse
import json
import math
import struct
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import GPT, Config  # noqa: E402

SEED = 20260829                       # the session date
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = dict(vocab_size=50257, n_layer=4, n_head=4, d_model=256, block_size=128, tie_weights=True)
LOG = []
FIGDIR = HERE / "figures"      # the notebook points this elsewhere so it never overwrites the full run

# Palette validated with the dataviz skill's checker (light surface, 2 slots, all checks pass).
BLUE, ORANGE, SURFACE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rule(n, title):
    say("")
    say("=" * 78)
    say(f"{n}. {title}")
    say("=" * 78)


# ------------------------------------------------------------------ data
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def corpus():
    p = HERE / "input.txt"
    if not p.exists():
        say(f"downloading {SHAKESPEARE}")
        urllib.request.urlretrieve(SHAKESPEARE, p)
    return p.read_text()


def token_splits():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(corpus()), dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:]


def windows(ids, B, T, g):
    """B random windows of T+1 tokens -> (inputs [B,T], targets [B,T])."""
    i = torch.randint(len(ids) - T - 1, (B,), generator=g)
    seq = torch.stack([ids[j:j + T + 1] for j in i]).to(DEV)
    return seq[:, :T].contiguous(), seq[:, 1:].contiguous()


def fresh_model(**over):
    torch.manual_seed(SEED)
    return GPT(Config(**{**CFG, **over})).to(DEV)


# ================================================================ 1. shapes
def req1(train_ids):
    rule(1, "Every tensor shape in one training step")
    model = fresh_model()
    cfg = model.cfg
    B, T, D, H, V = 4, 64, cfg.d_model, cfg.n_head, cfg.vocab_size
    Dh = D // H
    idx, tgt = windows(train_ids, B, T, torch.Generator().manual_seed(SEED))
    rows = []

    def add(name, t, meaning):
        shape = list(t.shape)
        rows.append(dict(tensor=name, shape=shape, dtype=str(t.dtype).replace("torch.", ""), meaning=meaning))
        say(f"  {name:<34} {str(tuple(shape)):<22} {meaning}")

    say(f"  B={B} sequences, T={T} positions, D={D} model width, H={H} heads, Dh={Dh}, V={V}")
    say("")
    add("idx", idx, f"[B={B} sequences, T={T} positions] token ids")
    add("targets", tgt, "[B, T] the id that should come next at each position")
    tok = model.wte(idx)
    add("tok_emb = wte[idx]", tok, f"[B, T, D={D}] one D-vector per token (a row lookup, no FLOPs)")
    pos = model.wpe(torch.arange(T, device=DEV))
    add("pos_emb = wpe[0..T-1]", pos, "[T, D] one vector per position, broadcast over B")
    x = tok + pos
    add("x = tok_emb + pos_emb", x, "[B, T, D] the residual stream entering block 0")

    blk = model.blocks[0]
    h = blk.n1(x)
    add("h = RMSNorm(x)", h, "[B, T, D] same shape; each token vector rescaled to unit RMS")
    qkv = blk.attn.qkv(h)
    add("qkv = h @ W_qkv^T", qkv, f"[B, T, 3D={3 * D}] query, key and value side by side")
    q, k, v = (t.view(B, T, H, Dh).transpose(1, 2) for t in qkv.split(D, dim=2))
    add("q", q, f"[B, H={H} heads, T, Dh={Dh}] D split into H heads of Dh")
    add("k", k, "[B, H, T, Dh] one key per head per position")
    add("v", v, "[B, H, T, Dh] one value per head per position")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    add("scores = q k^T / sqrt(Dh)", scores, "[B, H, T queries, T keys] every position scored against every other")
    causal = torch.ones(T, T, dtype=torch.bool, device=DEV).triu(1)
    attn = scores.masked_fill(causal, float("-inf")).softmax(-1)
    add("attn = softmax(masked scores)", attn, "[B, H, T, T] each row sums to 1; the future is masked to 0")
    y = attn @ v
    add("y = attn @ v", y, "[B, H, T, Dh] per head, a weighted mix of values")
    cat = y.transpose(1, 2).reshape(B, T, D)
    add("concat heads", cat, "[B, T, D] heads laid back side by side")
    a = blk.attn.proj(cat)
    add("W_O @ concat", a, "[B, T, D] the output projection; this is what mixes the heads")
    assert torch.allclose(a, blk.attn(h), atol=1e-5), "hand-unrolled attention disagrees with the module"
    x = x + a
    add("x = x + attn_out", x, "[B, T, D] residual add")
    h2 = blk.n2(x)
    gate, up = blk.ffn.gate(h2), blk.ffn.up(h2)
    add("gate(h), up(h)", gate, f"[B, T, F={gate.shape[-1]}] FFN width, 8/3 D rounded down to a multiple of 64")
    act = F.silu(gate) * up
    add("silu(gate) * up", act, "[B, T, F] the SwiGLU product")
    x = x + blk.ffn.down(act)
    add("x = x + down(...)", x, "[B, T, D] back to model width, residual add; block 0 done")
    for b in model.blocks[1:]:
        x = b(x)
    add(f"x after all {cfg.n_layer} blocks", x, "[B, T, D] the stream, unchanged in shape by every block")
    hN = model.norm_f(x)
    logits = model.lm_head(hN)
    add("logits = norm_f(x) @ W_head^T", logits, f"[B, T, V={V}] one score per vocabulary entry per position")
    assert torch.allclose(logits, model(idx), atol=1e-4), "hand-unrolled forward disagrees with model()"
    flat = logits.reshape(-1, V)
    add("logits.view(B*T, V)", flat, f"[B*T={B * T}, V] every position becomes one classification row")
    loss = F.cross_entropy(flat, tgt.reshape(-1))
    add("loss", loss, f"[] one scalar: the mean over all B*T={B * T} predictions")
    say(f"  loss = {loss.item():.4f}  (ln V = {math.log(V):.4f}: an untrained model should sit near it)")

    loss.backward()
    say("")
    say("  backward(): every gradient has exactly the shape of the weight it belongs to")
    grads = [
        ("wte.weight = lm_head.weight (tied)", model.wte.weight, "[V rows, D] one row per token; gets gradient from the lookup AND the head"),
        ("wpe.weight", model.wpe.weight, "[block_size=128, D] only the first T=64 rows got nonzero gradient"),
        ("blocks.0.attn.qkv.weight", blk.attn.qkv.weight, "[out=3D, in=D] nn.Linear stores [out, in]"),
        ("blocks.0.attn.proj.weight", blk.attn.proj.weight, "[out=D, in=D] W_O"),
        ("blocks.0.ffn.down.weight", blk.ffn.down.weight, "[out=D, in=F]"),
        ("blocks.0.n1.g", blk.n1.g, "[D] one gain per channel"),
    ]
    for name, p, meaning in grads:
        add(f"grad {name}", p.grad, meaning)
    unused = int((model.wpe.weight.grad[T:].abs().sum(-1) == 0).sum())
    say(f"  wpe rows with zero gradient: {unused} of {model.wpe.weight.shape[0]} (positions >= T were never used)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    st = opt.state[blk.attn.qkv.weight]
    say("")
    say("  optimizer.step(): AdamW keeps two tensors per weight, each the weight's shape")
    add("exp_avg (m), qkv.weight", st["exp_avg"], "[3D, D] running mean of the gradient")
    add("exp_avg_sq (v), qkv.weight", st["exp_avg_sq"], "[3D, D] running mean of the squared gradient")
    opt.zero_grad(set_to_none=True)
    say(f"  zero_grad(set_to_none=True): qkv.weight.grad is now {blk.attn.qkv.weight.grad}")

    n_all = model.n_params()
    n_emb = model.wte.weight.numel() + model.wpe.weight.numel()
    say(f"  parameters: {n_all:,} total, {n_all - n_emb:,} outside the embeddings")
    return dict(rows=rows, loss=loss.item(), ln_v=math.log(V), n_params=n_all,
                n_non_embedding=n_all - n_emb, wpe_unused_rows=unused, B=B, T=T)


# ======================================================= 2. one gradient by hand
def loss_on(model, idx, tgt):
    return F.cross_entropy(model(idx).reshape(-1, model.cfg.vocab_size).double(), tgt.reshape(-1))


def nudge(model, w, r, c, eps, idx, tgt):
    """Central and forward differences. Uses the step actually stored, not the one asked for:
    in fp32, w + 1e-6 is not exactly 1e-6 away from w."""
    with torch.no_grad():
        orig = w[r, c].clone()
        l0 = loss_on(model, idx, tgt).item()
        w[r, c] = orig + eps
        hp = (w[r, c].double() - orig.double()).item()
        lp = loss_on(model, idx, tgt).item()
        w[r, c] = orig - eps
        hm = (orig.double() - w[r, c].double()).item()
        lm = loss_on(model, idx, tgt).item()
        w[r, c] = orig
    return (lp - lm) / (hp + hm), (lp - l0) / hp


def _legacy_rmsnorm(self, x):
    """The Session 9 RMSNorm, verbatim: x.float() upcasts bf16 but downcasts fp64."""
    rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
    return (x.float() * rms).type_as(x) * self.g


def fp64_check(idx, tgt, where, legacy):
    from model import RMSNorm
    fixed = RMSNorm.forward
    if legacy:
        RMSNorm.forward = _legacy_rmsnorm
    try:
        model = fresh_model().double()
        w = model.blocks[1].ffn.up.weight
        loss_on(model, idx, tgt).backward()
        auto = w.grad[where[1], where[2]].item()
        central, _ = nudge(model, w, where[1], where[2], 1e-6, idx, tgt)
    finally:
        RMSNorm.forward = fixed
    rel = abs(central - auto) / abs(auto)
    return auto, central, rel


def req2(train_ids):
    rule(2, "One gradient, verified by hand")
    idx, tgt = windows(train_ids, 2, 32, torch.Generator().manual_seed(SEED + 2))
    where = ("blocks.1.ffn.up.weight", 5, 17)

    # First attempt, with the model exactly as Session 9 left it.
    a0, c0, r0 = fp64_check(idx, tgt, where, legacy=True)
    say(f"  first attempt, model as in Session 9: fp64 model, central difference, eps 1e-6")
    say(f"    backward() {a0:+.12f}   nudge {c0:+.12f}   relative error {r0:.2e}  <- they should agree to ~1e-9")
    say(f"  they do not. The error grows as eps shrinks, even in fp64 -- the signature of lost precision.")
    say(f"  cause: RMSNorm computed x.float(). For bf16 that is an upcast; for an fp64 model it is a silent")
    say(f"  DOWNCAST to fp32 inside every norm, so the 'fp64' loss only carried fp32's 7 digits.")
    say(f"  fix (model.py): promote to at least fp32 -- x.to(torch.promote_types(x.dtype, torch.float32)).")
    say("")
    out = {}
    for dtype in (torch.float64, torch.float32):
        model = fresh_model().to(dtype)
        w = model.blocks[1].ffn.up.weight
        model.zero_grad()
        loss_on(model, idx, tgt).backward()
        out[str(dtype)] = dict(model=model, w=w, auto=w.grad[where[1], where[2]].item())
    truth = out["torch.float64"]["auto"]
    say(f"  after the fix -- weight {where[0]}[{where[1]}, {where[2]}], batch 2 x 32 tokens")
    say(f"  backward() in fp64 : {truth:+.12f}   <- treated as ground truth")
    say(f"  backward() in fp32 : {out['torch.float32']['auto']:+.12f}")

    # the headline check: fp64, central difference, eps = 1e-6
    m64, w64 = out["torch.float64"]["model"], out["torch.float64"]["w"]
    central, forward = nudge(m64, w64, where[1], where[2], 1e-6, idx, tgt)
    agree = -int(math.floor(math.log10(abs(central - truth) / abs(truth))))
    say(f"  nudge  +/-1e-6 fp64 : {central:+.12f}   (central difference)")
    say(f"  |nudge - backward|  : {abs(central - truth):.2e}   relative {abs(central - truth) / abs(truth):.2e}"
        f"   -> agree to {agree} significant figures")

    # the part worth understanding: the same check, done carelessly
    say("")
    say("  the same check across nudge sizes -- relative error against the fp64 backward()")
    say(f"  {'eps':>8} | {'fp64 central':>12} {'fp64 forward':>12} | {'fp32 central':>12} {'fp32 forward':>12}")
    sweep = []
    m32, w32 = out["torch.float32"]["model"], out["torch.float32"]["w"]
    for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        c64, f64 = nudge(m64, w64, where[1], where[2], e, idx, tgt)
        c32, f32 = nudge(m32, w32, where[1], where[2], e, idx, tgt)
        rel = [abs(v - truth) / abs(truth) for v in (c64, f64, c32, f32)]
        sweep.append(dict(eps=e, fp64_central=rel[0], fp64_forward=rel[1], fp32_central=rel[2], fp32_forward=rel[3]))
        say(f"  {e:>8.0e} | {rel[0]:>12.2e} {rel[1]:>12.2e} | {rel[2]:>12.2e} {rel[3]:>12.2e}")
    best32 = min(sweep, key=lambda r: r["fp32_central"])
    say(f"  fp32 is best at eps={best32['eps']:.0e} ({best32['fp32_central']:.1e}); smaller nudges get worse, "
        f"because the loss change drops below fp32's ~7 digits and the difference is mostly rounding")
    return dict(weight=f"{where[0]}[{where[1]},{where[2]}]", autograd_fp64=truth,
                autograd_fp32=out["torch.float32"]["auto"], nudge_fp64=central,
                abs_err=abs(central - truth), rel_err=abs(central - truth) / abs(truth),
                agree_sig_figs=agree, sweep=sweep, best_fp32_eps=best32["eps"],
                legacy=dict(autograd=a0, nudge=c0, rel_err=r0))


# ============================================== 3 + 4. accumulation, grad norm
def micro(ids, g, B, L, T):
    """B sequences with only the first L positions counted; the rest padded out of the loss."""
    idx, tgt = windows(ids, B, T, g)
    tgt[:, L:] = -100
    return idx, tgt


def heldout(model, val, T=128, n=32):
    """Token-weighted loss on fixed full-length sequences, split by position."""
    g = torch.Generator().manual_seed(SEED + 99)
    idx, tgt = windows(val, n, T, g)
    model.eval()
    with torch.no_grad(), torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
        per = F.cross_entropy(model(idx).float().reshape(-1, model.cfg.vocab_size),
                              tgt.reshape(-1), reduction="none").view(n, T)
    model.train()
    return per.mean().item(), per[:, :16].mean().item(), per[:, 64:].mean().item()


def train_accum(train_ids, val, mode, steps, lengths_mode="mixed", lr=1e-3, clip=1.0, eval_every=25):
    """K micro-batches per step. mode = 'correct' (sum of token losses / all tokens)
    or 'buggy' (mean of per-micro-batch means). Same init and same data for both."""
    model = fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=DEV == "cuda")
    g = torch.Generator().manual_seed(SEED + 3)            # identical data order across modes
    K, B, T, V = 4, 8, 128, model.cfg.vocab_size
    choices = [8, 16, 32, 64, 128]
    log = dict(step=[], reported=[], true=[], gnorm=[], eval_step=[], eval=[], eval_early=[], eval_late=[])
    for step in range(steps):
        if lengths_mode == "mixed":
            Ls = [choices[int(torch.randint(len(choices), (1,), generator=g))] for _ in range(K)]
        else:
            Ls = [T] * K
        batches = [micro(train_ids, g, B, L, T) for L in Ls]
        n_tok = [int((t != -100).sum()) for _, t in batches]
        N = sum(n_tok)
        sums, means = [], []
        for (idx, tgt), n in zip(batches, n_tok):
            with torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
                logits = model(idx)
            s = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1), reduction="sum")
            part = s / N if mode == "correct" else (s / n) / K
            part.backward()
            sums.append(s.item())
            means.append(s.item() / n)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf")).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        true = sum(sums) / N
        log["step"].append(step)
        log["true"].append(true)
        log["reported"].append(true if mode == "correct" else sum(means) / K)
        log["gnorm"].append(gn)
        if step % eval_every == 0 or step == steps - 1:
            a, e, l = heldout(model, val)
            log["eval_step"].append(step)
            log["eval"].append(a)
            log["eval_early"].append(e)
            log["eval_late"].append(l)
    return model, log


def equal_length_control(train_ids):
    """Same model state, one step's micro-batches all at full length: the two losses must coincide."""
    model = fresh_model()
    g = torch.Generator().manual_seed(SEED + 4)
    batches = [micro(train_ids, g, 8, 128, 128) for _ in range(4)]
    V = model.cfg.vocab_size
    grads, losses = {}, {}
    for mode in ("correct", "buggy"):
        model.zero_grad(set_to_none=True)
        N = sum(int((t != -100).sum()) for _, t in batches)
        total = 0.0
        for idx, tgt in batches:
            n = int((tgt != -100).sum())
            s = F.cross_entropy(model(idx).reshape(-1, V), tgt.reshape(-1), reduction="sum")
            part = s / N if mode == "correct" else (s / n) / 4
            part.backward()
            total += part.item()
        losses[mode] = total
        grads[mode] = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
    rel = ((grads["correct"] - grads["buggy"]).norm() / grads["correct"].norm()).item()
    return dict(loss_correct=losses["correct"], loss_buggy=losses["buggy"],
                loss_abs_diff=abs(losses["correct"] - losses["buggy"]), grad_rel_diff=rel)


def req3(train_ids, val, steps):
    rule(3, "Gradient accumulation, broken on purpose")
    say("  4 micro-batches of 8 sequences per step; each micro-batch keeps 8, 16, 32, 64 or 128 of its 128")
    say("  positions in the loss (the rest are padding). Same init, same data order, two ways to combine:")
    say("    correct : sum every token's loss, divide by all valid tokens in the step")
    say("    buggy   : average each micro-batch, then average the averages")
    t0 = time.time()
    _, good = train_accum(train_ids, val, "correct", steps)
    _, bad = train_accum(train_ids, val, "buggy", steps)
    say(f"  two runs of {steps} steps in {time.time() - t0:.0f}s")

    gap = [abs(r - t) / t for r, t in zip(bad["reported"], bad["true"])]
    worst = max(range(len(gap)), key=gap.__getitem__)
    say(f"  buggy run, reported vs true loss on the same batch: mean gap {100 * sum(gap) / len(gap):.2f}%, "
        f"worst {100 * gap[worst]:.2f}% at step {worst} ({bad['reported'][worst]:.3f} vs {bad['true'][worst]:.3f})")
    say("")
    say(f"  held-out loss (token-weighted, full-length sequences)")
    say(f"  {'step':>6} {'correct':>9} {'buggy':>9} | {'early pos 0-15':>16} | {'late pos 64-127':>16}")
    for i, s in enumerate(good["eval_step"]):
        if i % 4 == 0 or i == len(good["eval_step"]) - 1:
            say(f"  {s:>6} {good['eval'][i]:>9.4f} {bad['eval'][i]:>9.4f} | "
                f"{good['eval_early'][i]:>7.3f} {bad['eval_early'][i]:>7.3f} | "
                f"{good['eval_late'][i]:>7.3f} {bad['eval_late'][i]:>7.3f}")
    ctl = equal_length_control(train_ids)
    say("")
    say(f"  control, all micro-batches full length: losses {ctl['loss_correct']:.6f} vs {ctl['loss_buggy']:.6f}"
        f" (diff {ctl['loss_abs_diff']:.1e}), gradients differ by {ctl['grad_rel_diff']:.1e} relative")
    say("  -> the bug is invisible whenever the token counts are equal, which is exactly how it hid.")
    fig = plot_accum(good, bad)
    return dict(steps=steps, correct=good, buggy=bad, mean_reported_gap=sum(gap) / len(gap),
                worst_gap=gap[worst], worst_gap_step=worst, control=ctl, figure=fig)


def robust_z(xs, i, w=20):
    win = sorted(xs[max(0, i - w):i])
    if len(win) < 8:
        return 0.0
    med = win[len(win) // 2]
    mad = sorted(abs(x - med) for x in win)[len(win) // 2] or 1e-12
    return (xs[i] - med) / (1.4826 * mad)


GN_Z, LOSS_QUIET, LOSS_Z, HORIZON, SKIP = 4.0, 2.0, 3.0, 10, 50


def lead_events(gn, loss):
    """Steps where the grad norm jumps UP out of its recent range while the loss is still quiet,
    and the loss then jumps UP within HORIZON steps. Recent range = rolling median and robust
    MAD over the previous 20 steps. The first SKIP steps are ignored: everything moves then."""
    events = []
    for s in range(SKIP, len(gn) - HORIZON):
        zg, zl = robust_z(gn, s), robust_z(loss, s)
        if zg >= GN_Z and zl < LOSS_QUIET:
            for d in range(1, HORIZON + 1):
                zlater = robust_z(loss, s + d)
                if zlater >= LOSS_Z:
                    events.append(dict(step=s, lead=d, z_gnorm=zg, z_loss_at_step=zl, z_loss_later=zlater))
                    break
    return events


def base_rates(gn, loss):
    """Is a loss spike after a grad-norm spike more than chance? Compare the rate after grad-norm
    spikes against the rate in every 10-step window of the run."""
    spikes = [s for s in range(SKIP, len(gn) - HORIZON) if robust_z(gn, s) >= GN_Z]
    lz = [robust_z(loss, t) for t in range(len(loss))]

    def followed(s):
        return any(lz[s + d] >= LOSS_Z for d in range(1, HORIZON + 1))
    after = sum(followed(s) for s in spikes) / len(spikes) if spikes else float("nan")
    windows_ = range(SKIP, len(gn) - HORIZON)
    base = sum(followed(s) for s in windows_) / len(windows_)
    return dict(gnorm_spikes=len(spikes), p_loss_spike_after_gnorm_spike=after, p_loss_spike_any_window=base)


def train_plain(train_ids, lr, clip, steps, B=16, T=128):
    """A plain loop, full-length batches, logging the pre-clip grad norm and the loss every step."""
    model = fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=DEV == "cuda")
    g = torch.Generator().manual_seed(SEED + 7)
    V = model.cfg.vocab_size
    gn, ls = [], []
    for _ in range(steps):
        idx, tgt = windows(train_ids, B, T, g)
        with torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(model(idx).float().reshape(-1, V), tgt.reshape(-1))
        loss.backward()
        # clip_grad_norm_ returns the norm BEFORE clipping, which is the one worth logging
        gn.append(torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf")).item())
        opt.step()
        opt.zero_grad(set_to_none=True)
        ls.append(loss.item())
    return gn, ls


def req4(train_ids, steps):
    rule(4, "The grad norm, every step -- and a step where it moved first")
    say(f"  rule: grad norm >= {GN_Z:g} robust-sigma ABOVE its last 20 steps while the loss is < {LOSS_QUIET:g}")
    say(f"        (still quiet), then the loss rises >= {LOSS_Z:g} robust-sigma within {HORIZON} steps.")
    say(f"        First {SKIP} steps skipped. Full-length batches, so per-step loss noise is not batch shape.")
    ladder = [(1e-3, 1.0, "lr 1e-3, clip 1.0 (the normal setting)"),
              (3e-3, 0.0, "lr 3e-3, no clipping"),
              (6e-3, 0.0, "lr 6e-3, no clipping"),
              (1e-2, 0.0, "lr 1e-2, no clipping")]
    tried = []
    for lr, clip, label in ladder:
        gn, ls = train_plain(train_ids, lr, clip, steps)
        ev = lead_events(gn, ls)
        br = base_rates(gn, ls)
        tried.append(dict(label=label, events=len(ev), final_loss=ls[-1], max_gnorm=max(gn), **br))
        say(f"  {label:<40} final loss {ls[-1]:.3f}  max grad norm {max(gn):8.2f}  "
            f"grad-norm spikes {br['gnorm_spikes']:>3}  lead events {len(ev)}")
        if ev:
            break
    if not ev:
        say("  no lead event on any rung of the ladder.")
        return dict(found=False, ladder=tried)
    e = ev[0]
    s = e["step"]
    br = tried[-1]
    say("")
    say(f"  found in: {label}")
    say(f"  step {s}: grad norm z={e['z_gnorm']:+.1f} while the loss sat at z={e['z_loss_at_step']:+.1f};"
        f" the loss followed {e['lead']} step(s) later at z={e['z_loss_later']:+.1f}")
    for t in range(max(0, s - 3), min(len(gn), s + e["lead"] + 3)):
        mark = "   <- grad norm moves" if t == s else ("   <- loss moves" if t == s + e["lead"] else "")
        say(f"    step {t:>4}  grad norm {gn[t]:>8.3f}  loss {ls[t]:.4f}{mark}")
    say(f"  is it more than chance? P(loss spike within {HORIZON} steps | grad-norm spike) = "
        f"{br['p_loss_spike_after_gnorm_spike']:.2f}, against {br['p_loss_spike_any_window']:.2f} for any {HORIZON}-step window")
    fig = plot_lead(gn, ls, s, s + e["lead"], label)
    return dict(found=True, source=label, events=ev, first=e, ladder=tried, gnorm=gn, loss=ls, figure=fig)


# ================================================================ 5. MFU
def matmul_peak(dtype=torch.bfloat16, n=8192, reps=10, windows_=5):
    """Best of several timed windows after a warm-up long enough for the clocks to settle.
    A laptop GPU boosts and throttles, so one short measurement undershoots."""
    a = torch.randn(n, n, device=DEV, dtype=dtype)
    b = torch.randn(n, n, device=DEV, dtype=dtype)
    t_end = time.time() + 2.0
    while time.time() < t_end:
        a @ b
    torch.cuda.synchronize()
    best = 0.0
    for _ in range(windows_):
        t0 = time.time()
        for _ in range(reps):
            a @ b
        torch.cuda.synchronize()
        best = max(best, 2 * n ** 3 * reps / (time.time() - t0))
    del a, b
    return best


def flops_per_token(model, T):
    """6 x (weights that take part in a matmul) + attention's own score/value matmuls.
    The embedding lookup costs nothing; the tied head is a real V x D matmul, so it counts."""
    cfg = model.cfg
    n_matmul = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    attn = 12 * cfg.n_layer * T * cfg.d_model          # QK^T and AV, forward + backward, no causal discount
    return 6 * n_matmul + attn, n_matmul


def time_steps(model, B, T, train_ids, amp, compiled, steps=30, warm=10):
    step_model = torch.compile(model) if compiled else model
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    g = torch.Generator().manual_seed(SEED + 5)
    V = model.cfg.vocab_size
    batches = [windows(train_ids, B, T, g) for _ in range(steps + warm)]
    for i, (idx, tgt) in enumerate(batches):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            loss = F.cross_entropy(step_model(idx).float().reshape(-1, V), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return B * T * steps / (time.time() - t0)


def profile_breakdown(train_ids):
    from torch.profiler import ProfilerActivity, profile
    model = fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    g = torch.Generator().manual_seed(SEED + 6)
    V = model.cfg.vocab_size
    batches = [windows(train_ids, 16, 128, g) for _ in range(8)]

    def step(idx, tgt):
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(idx).float().reshape(-1, V), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    for b in batches[:3]:
        step(*b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for b in batches[3:]:
            step(*b)
        torch.cuda.synchronize()
    buckets = {}
    total = 0.0
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if t <= 0 or e.key.startswith("ProfilerStep"):
            continue
        k = e.key.lower()
        # attention first: the memory-efficient kernel is named fmha_cutlass*, which the
        # matmul test below would otherwise claim
        if "flash" in k or "attention" in k or "efficient" in k or "fmha" in k:
            b = "attention kernels"
        elif "gemm" in k or "matmul" in k or "cutlass" in k or "sm80" in k or "sm89" in k or "ampere" in k:
            b = "matmul kernels"
        elif "softmax" in k or "cross_entropy" in k or "log_softmax" in k or "nll" in k:
            b = "softmax / cross-entropy over V"
        elif "adam" in k or "foreach" in k or "multi_tensor" in k or "clip" in k:
            b = "optimizer + grad clipping"
        elif "embedding" in k or "index" in k or "gather" in k or "scatter" in k:
            b = "embedding lookup / scatter"
        else:
            b = "elementwise, norms, casts, copies"
        buckets[b] = buckets.get(b, 0.0) + t
        total += t
    return {k: v / total for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}


def req5(train_ids, quick):
    rule(5, "MFU, measured honestly")
    if DEV != "cuda":
        say("  no GPU: skipped")
        return dict(skipped=True)
    name = torch.cuda.get_device_name()
    peak_bf16 = matmul_peak(torch.bfloat16)
    peak_fp32 = matmul_peak(torch.float32)
    say(f"  {name}")
    say(f"  measured peak, one 8192^3 matmul: bf16 {peak_bf16 / 1e12:.1f} TFLOP/s, fp32 {peak_fp32 / 1e12:.1f} TFLOP/s")
    say("  every MFU below divides by the bf16 figure: a laptop part's datasheet peak depends on its power")
    say("  limit, so the denominator is what this card actually sustains on the friendliest workload there is")
    variants = [
        dict(label="fp32, eager", over={}, B=16, T=128, amp=False, compiled=False),
        dict(label="bf16 autocast, eager", over={}, B=16, T=128, amp=True, compiled=False),
        dict(label="bf16, torch.compile", over={}, B=16, T=128, amp=True, compiled=True),
        dict(label="bf16, compile, 4x wider model (D=768, 8 layers)", over=dict(d_model=768, n_layer=8, n_head=12),
             B=8, T=256, amp=True, compiled=True),
    ]
    if quick:
        variants = [v for v in variants if not v["compiled"]]
    rows = []
    say("")
    say(f"  {'variant':<48} {'tok/s':>9} {'6N (brief)':>11} {'exact':>7}")
    for v in variants:
        torch.cuda.empty_cache()
        model = fresh_model(**{**v["over"], "block_size": max(128, v["T"])})
        fpt, n_matmul = flops_per_token(model, v["T"])
        N = model.n_params()
        try:
            tps = time_steps(model, v["B"], v["T"], train_ids, v["amp"], v["compiled"])
        except Exception as ex:                       # torch.compile can fail on odd installs
            say(f"  {v['label']:<48} failed: {type(ex).__name__}")
            continue
        # one denominator for every row: the best this card can do (bf16 tensor cores).
        # Dividing an fp32 run by an fp32 peak would flatter it -- MFU asks what share of the
        # machine you paid for is doing model maths, and fp32 simply leaves most of it idle.
        peak = peak_bf16
        mfu6n, mfu = 6 * N * tps / peak, fpt * tps / peak
        rows.append(dict(label=v["label"], tokens_per_s=tps, params=N, matmul_params=n_matmul,
                         flops_per_token=fpt, mfu_6n=mfu6n, mfu=mfu, peak=peak))
        say(f"  {v['label']:<48} {tps:>9,.0f} {100 * mfu6n:>10.1f}% {100 * mfu:>6.1f}%")
        del model
    prof = profile_breakdown(train_ids)
    say("")
    say("  where the GPU time goes (bf16 eager, the base model; share of CUDA kernel time):")
    for k, v in prof.items():
        say(f"    {k:<38} {100 * v:5.1f}%")
    return dict(device=name, peak_bf16=peak_bf16, peak_fp32=peak_fp32, variants=rows, profile=prof)


# ============================================================ 6. 0.1 in bits
def encode(x: Fraction, ebits, mbits, bias):
    """Round-to-nearest-even encoding of a positive normal number, done in exact arithmetic."""
    e = 0
    while x >= 2:
        x /= 2
        e += 1
    while x < 1:
        x *= 2
        e -= 1
    frac = (x - 1) * (1 << mbits)                     # the mantissa, before rounding
    lo = int(frac)
    rem = frac - lo
    m = lo + (1 if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and lo % 2) else 0)
    if m == 1 << mbits:                               # rounding carried into the exponent
        m, e = 0, e + 1
    field = e + bias
    assert 0 < field < (1 << ebits) - (0 if ebits == 4 else 1), "0.1 is a normal number in all three"
    value = (1 + Fraction(m, 1 << mbits)) * (Fraction(2) ** e)
    return dict(sign="0", exponent=format(field, f"0{ebits}b"), mantissa=format(m, f"0{mbits}b"),
                unbiased_exp=e, exp_field=field, mantissa_int=m, mantissa_exact=float(frac),
                value=float(value), value_exact=str(value), rel_err=float(abs(value - Fraction(1, 10)) / Fraction(1, 10)))


def req6():
    rule(6, "0.1 in fp32, bf16 and fp8 E4M3, by hand")
    x = Fraction(1, 10)
    say("  0.1 = 1.6 x 2^-4, so every format stores exponent -4 and has to approximate the 0.6.")
    say("  0.6 in binary is 0.1001 1001 1001 ... -- it repeats forever, so every format must round.")
    fmts = [("fp32", 8, 23, 127), ("bf16", 8, 7, 127), ("fp8 E4M3", 4, 3, 7)]
    out = {}
    for name, eb, mb, bias in fmts:
        r = encode(x, eb, mb, bias)
        bits = r["sign"] + r["exponent"] + r["mantissa"]
        r["bits"] = bits
        r["hex"] = f"0x{int(bits, 2):0{len(bits) // 4}X}"
        out[name] = r
        say("")
        say(f"  {name}: 1 sign + {eb} exponent + {mb} mantissa, bias {bias}")
        say(f"    exponent field = -4 + {bias} = {r['exp_field']} = {r['exponent']}")
        say(f"    mantissa = 0.6 x 2^{mb} = {r['mantissa_exact']:.4f} -> rounds to {r['mantissa_int']} = {r['mantissa']}")
        say(f"    bits  {r['sign']} {r['exponent']} {r['mantissa']}   ({r['hex']})")
        say(f"    value (1 + {r['mantissa_int']}/2^{mb}) x 2^-4 = {r['value']:.12g}   error {100 * r['rel_err']:.4g}%")

    # check the hand arithmetic against what the hardware formats actually produce
    fp32_bits = format(struct.unpack(">I", struct.pack(">f", 0.1))[0], "032b")
    bf16_bits = format(torch.tensor(0.1, dtype=torch.float64).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF, "016b")
    checks = {"fp32": fp32_bits, "bf16": bf16_bits}
    if hasattr(torch, "float8_e4m3fn"):
        checks["fp8 E4M3"] = format(torch.tensor(0.1).to(torch.float8_e4m3fn).view(torch.uint8).item(), "08b")
    say("")
    for k, v in checks.items():
        ok = v == out[k]["bits"]
        out[k]["torch_bits"] = v
        out[k]["matches_torch"] = ok
        say(f"  {k:<9} by hand {out[k]['bits']}  torch {v}  {'MATCH' if ok else 'MISMATCH'}")
    return out


# ================================================================ figures
def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _ema(xs, a=0.9):
    out, m = [], xs[0]
    for x in xs:
        m = a * m + (1 - a) * x
        out.append(m)
    return out


def plot_accum(good, bad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), facecolor=SURFACE)
    s = bad["step"]
    # Plotted as the gap itself: two loss curves 1.5% apart on an 11-to-5 scale are one line.
    gap = [100 * (r - t) / t for r, t in zip(bad["reported"], bad["true"])]
    a1.plot(s, gap, color=ORANGE, lw=0.8, alpha=0.45)
    a1.plot(s, _ema(gap), color=ORANGE, lw=1.8)
    a1.axhline(0, color=INK2, lw=1)
    worst = max(range(len(gap)), key=lambda i: abs(gap[i]))
    a1.annotate(f"step {s[worst]}: {gap[worst]:+.1f}%", (s[worst], gap[worst]), textcoords="offset points",
                xytext=(-110, 0), color=INK, fontsize=9, va="center")
    a1.text(s[-1], 0, " 0 = correct average", color=INK2, fontsize=8, va="bottom", ha="right")
    _style(a1, "Reported minus true loss, same run, same batches (thin: every step; thick: EMA 0.9)",
           "gap, % of the true loss")
    a2.plot(good["eval_step"], good["eval"], color=BLUE, lw=1.6, marker="o", ms=3, label="trained with the correct average")
    a2.plot(bad["eval_step"], bad["eval"], color=ORANGE, lw=1.6, marker="o", ms=3, label="trained with the average of averages")
    a2.text(good["eval_step"][-1], good["eval"][-1], f"  {good['eval'][-1]:.3f}", color=INK, va="top", fontsize=9)
    a2.text(bad["eval_step"][-1], bad["eval"][-1], f"  {bad['eval'][-1]:.3f}", color=INK, va="bottom", fontsize=9)
    _style(a2, "Held-out loss of the two runs (full-length sequences, token-weighted)", "held-out loss")
    a2.set_xlabel("step", color=INK2, fontsize=9)
    a2.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = FIGDIR / "accumulation_gap.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return str(p.relative_to(HERE))


def plot_lead(gn, loss, s_gn, s_loss, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = max(0, s_gn - 40), min(len(gn), s_loss + 40)
    xs = list(range(lo, hi))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, facecolor=SURFACE)
    a1.plot(xs, gn[lo:hi], color=BLUE, lw=1.6)
    a2.plot(xs, loss[lo:hi], color=ORANGE, lw=1.6)
    for ax in (a1, a2):
        ax.axvline(s_gn, color=INK2, lw=1, ls="--")
        ax.axvline(s_loss, color=INK2, lw=1, ls=":")
    a1.text(s_gn, max(gn[lo:hi]), f" step {s_gn}: grad norm moves", color=INK, fontsize=9, va="top")
    a2.text(s_loss, max(loss[lo:hi]), f" step {s_loss}: loss moves", color=INK, fontsize=9, va="top")
    _style(a1, "Grad norm (pre-clip), every step", "grad norm")
    _style(a2, "Training loss, every step", "loss")
    a2.set_xlabel(f"step -- {source}", color=INK2, fontsize=9)
    fig.tight_layout()
    p = FIGDIR / "gradnorm_leads_loss.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return str(p.relative_to(HERE))


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw figures/ from results.json, no training")
    args = ap.parse_args()
    if args.replot:
        R = json.loads((HERE / "results.json").read_text())
        plot_accum(R["req3"]["correct"], R["req3"]["buggy"])
        e = R["req4"]["first"]
        plot_lead(R["req4"]["gnorm"], R["req4"]["loss"], e["step"], e["step"] + e["lead"], R["req4"]["source"])
        print("figures redrawn")
        return
    steps = 120 if args.quick else 400
    torch.manual_seed(SEED)
    t0 = time.time()
    say(f"ERA V5 Session 10 -- training loop | seed {SEED} | device {DEV}"
        + (f" ({torch.cuda.get_device_name()})" if DEV == "cuda" else "") + f" | torch {torch.__version__}")
    train_ids, val = token_splits()
    say(f"tiny Shakespeare, GPT-2 BPE: {len(train_ids):,} train / {len(val):,} val tokens")
    R = {}
    R["req1"] = req1(train_ids)
    R["req2"] = req2(train_ids)
    R["req3"] = req3(train_ids, val, steps)
    R["req4"] = req4(train_ids, steps)
    R["req5"] = req5(train_ids, args.quick)
    R["req6"] = req6()
    R["meta"] = dict(seed=SEED, torch=torch.__version__, device=DEV,
                     gpu=torch.cuda.get_device_name() if DEV == "cuda" else None,
                     steps=steps, quick=args.quick, config=CFG, seconds=round(time.time() - t0))
    say("")
    say(f"done in {time.time() - t0:.0f}s")
    (HERE / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (HERE / "run.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
