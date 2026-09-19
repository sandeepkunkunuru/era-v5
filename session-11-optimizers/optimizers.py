"""ERA V5 · Session 11 — optimizers and learning-rate schedules, measured rather than asserted.

One function per requirement of the assignment:

    req1  Adam by hand: one weight, five gradients; m, v, m_hat, v_hat and the step, checked against PyTorch
    req2  bias correction off: the first twenty steps both ways, and when the difference stops mattering
    req3  the update-to-weight ratio for every layer, and the step at which warmup stops changing it
    req4  cosine against WSD, 300-step schedules stopped at step 200 -- each side tuned before comparing
    req5  a learning-rate sweep at widths 256, 512 and 1,024, and a call for width 4,096

    python optimizers.py          # full run: writes run.log, results.json, figures/
    python optimizers.py --quick  # same structure, fewer points
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import GPT, Config  # noqa: E402

SEED = 20260905                          # the session date
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = dict(vocab_size=50257, n_layer=4, n_head=4, d_model=256, block_size=128, tie_weights=True)
B, T = 16, 128
LOG = []

# Palette validated with the dataviz skill's checker. Aqua sits below 3:1 on the surface, so
# every series carrying it is direct-labelled and the numbers are also given as a table.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


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


def token_splits():
    import tiktoken
    p = HERE / "input.txt"
    if not p.exists():
        say(f"downloading {SHAKESPEARE}")
        urllib.request.urlretrieve(SHAKESPEARE, p)
    ids = torch.tensor(tiktoken.get_encoding("gpt2").encode(p.read_text()), dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:]


def windows(ids, b, t, g):
    i = torch.randint(len(ids) - t - 1, (b,), generator=g)
    seq = torch.stack([ids[j:j + t + 1] for j in i]).to(DEV)
    return seq[:, :t].contiguous(), seq[:, 1:].contiguous()


def fresh_model(**over):
    torch.manual_seed(SEED)
    return GPT(Config(**{**CFG, **over})).to(DEV)


VAL = {}


def eval_loss(model, val, n=128):
    """Token-weighted loss on a fixed held-out set (the same windows for every model)."""
    if "batch" not in VAL:
        VAL["batch"] = windows(val, n, T, torch.Generator().manual_seed(SEED + 99))
    idx, tgt = VAL["batch"]
    model.eval()
    tot = 0.0
    with torch.no_grad(), torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
        for i in range(0, n, 32):
            lg = model(idx[i:i + 32]).float()
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[i:i + 32].reshape(-1), reduction="sum").item()
    model.train()
    return tot / (n * T)


# ------------------------------------------------------------------ optimizer
class AdamWx(torch.optim.Optimizer):
    """AdamW written out, with bias correction switchable. Checked against torch.optim.AdamW
    below before it is trusted with anything (requirement 2)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.bias_correction = bias_correction

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t, m, v, g = st["t"], st["m"], st["v"], p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = (1 - b1 ** t, 1 - b2 ** t) if self.bias_correction else (1.0, 1.0)
                p.mul_(1 - group["lr"] * group["weight_decay"])          # decoupled decay
                p.addcdiv_(m / bc1, (v / bc2).sqrt().add_(group["eps"]), value=-group["lr"])


def param_groups(model, wd=0.1):
    """Decay the matrices; leave norm gains out, as the brief says (Section 7)."""
    decay, no = [], []
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no, weight_decay=0.0)]


def train(train_ids, val, peak, sched, steps, over=None, bias_correction=True, custom=False,
          log_ratio=False, eval_at=(), data_seed=1, stop_at=None, model=None, opt=None, start=0):
    """One training run. sched(step) -> multiplier on the peak learning rate."""
    model = model or fresh_model(**(over or {}))
    if opt is None:
        opt = (AdamWx(param_groups(model), lr=peak, bias_correction=bias_correction) if custom
               else torch.optim.AdamW(param_groups(model), lr=peak, fused=DEV == "cuda"))
    g = torch.Generator().manual_seed(SEED + data_seed)
    for _ in range(start):                           # keep the data stream aligned when resuming
        torch.randint(len(train_ids) - T - 1, (B,), generator=g)
    V = model.cfg.vocab_size
    names = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]
    out = dict(loss=[], lr=[], evals={}, ratios={n: [] for n, _ in names} if log_ratio else None)
    end = stop_at if stop_at is not None else steps
    for step in range(start, end):
        for grp in opt.param_groups:
            grp["lr"] = peak * sched(step)
        idx, tgt = windows(train_ids, B, T, g)
        with torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(model(idx).float().reshape(-1, V), tgt.reshape(-1))
        if not torch.isfinite(loss):
            out["diverged_at"] = step
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if log_ratio:
            before = {n: p.detach().clone() for n, p in names}
        opt.step()
        opt.zero_grad(set_to_none=True)
        if log_ratio:
            for n, p in names:
                out["ratios"][n].append(((p.detach() - before[n]).norm() / before[n].norm()).item())
        out["loss"].append(loss.item())
        out["lr"].append(peak * sched(step))
        if (step + 1) in eval_at:
            out["evals"][step + 1] = eval_loss(model, val)
    out["model"], out["opt"] = model, opt
    return out


# ================================================================ 1. Adam by hand
BRIEF_TABLE = {   # Section 6 of the brief, as printed (w0 = 1.0, eta = 0.001)
    "m": [0.0500, 0.0850, 0.1365, 0.1678, 0.2061],
    "v": [0.000250, 0.000410, 0.000769, 0.000971, 0.001273],
    "m_hat": [0.5000, 0.4474, 0.5037, 0.4881, 0.5032],
    "v_hat": [0.2500, 0.2050, 0.2567, 0.2431, 0.2550],
    "step": [-0.001000, -0.000988, -0.000994, -0.000990, -0.000996],
    "w": [0.999000, 0.998012, 0.997018, 0.996028, 0.995031],
}


def adam_by_hand(grads, w, lr, b1=0.9, b2=0.999, eps=1e-8, bias_correction=True):
    m = v = 0.0
    rows = []
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t) if bias_correction else m
        vh = v / (1 - b2 ** t) if bias_correction else v
        step = -lr * mh / (math.sqrt(vh) + eps)
        w += step
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=mh, v_hat=vh, step=step, w=w))
    return rows


def adam_torch(grads, w0, lr, dtype):
    w = torch.tensor([w0], dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    rows = []
    for g in grads:
        w.grad = torch.tensor([g], dtype=dtype)
        before = w.item()
        opt.step()
        st = opt.state[w]
        t = int(st["step"].item())
        m, v = st["exp_avg"].item(), st["exp_avg_sq"].item()
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=m / (1 - 0.9 ** t), v_hat=v / (1 - 0.999 ** t),
                         step=w.item() - before, w=w.item()))
    return rows


def req1():
    rule(1, "Adam by hand: one weight, five gradients")
    grads, w0, lr = [0.5, 0.4, 0.6, 0.45, 0.55], 1.0, 1e-3
    hand = adam_by_hand(grads, w0, lr)
    say(f"  w0 = {w0}, eta = {lr}, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 -- the brief's Section 6 example")
    say("")
    say(f"  {'t':>2} {'g':>5} {'m':>10} {'v':>12} {'m_hat':>10} {'v_hat':>10} {'step':>13} {'w':>11}")
    for r in hand:
        say(f"  {r['t']:>2} {r['g']:>5.2f} {r['m']:>10.6f} {r['v']:>12.9f} {r['m_hat']:>10.6f} "
            f"{r['v_hat']:>10.6f} {r['step']:>+13.9f} {r['w']:>11.9f}")
    keys = ["m", "v", "m_hat", "v_hat", "step", "w"]
    agree = {}
    say("")
    for dtype in (torch.float64, torch.float32):
        tr = adam_torch(grads, w0, lr, dtype)
        diffs = {k: max(abs(a[k] - b[k]) for a, b in zip(hand, tr)) for k in keys}
        dec = {k: (99 if d == 0 else int(-math.floor(math.log10(d)))) for k, d in diffs.items()}
        agree[str(dtype)] = dict(max_abs_diff=diffs, decimals=dec)
        say(f"  torch.optim.Adam in {str(dtype).replace('torch.', ''):<8} worst |hand - torch| per quantity: "
            + ", ".join(f"{k} {diffs[k]:.1e}" for k in keys))
    # and against the brief's own printed table, at the precision it printed
    say("")
    mismatches = []
    for k, vals in BRIEF_TABLE.items():
        for r, printed in zip(hand, vals):
            places = len(f"{printed:f}".rstrip("0").split(".")[1]) if "." in f"{printed:f}" else 0
            places = max(places, len(str(printed).split(".")[1]) if "." in str(printed) else 0)
            if abs(round(r[k], places) - printed) > 10 ** -places / 2 + 1e-15:
                mismatches.append(dict(quantity=k, t=r["t"], printed=printed, computed=r[k]))
    say(f"  against the brief's printed table: {30 - len(mismatches)} of 30 values reproduce"
        + ("" if not mismatches else "; differ: " + ", ".join(
            f"{x['quantity']}@t={x['t']} printed {x['printed']} vs {x['computed']:.6f}" for x in mismatches)))
    return dict(hand=hand, agreement=agree, brief_mismatches=mismatches)


# ====================================================== 2. bias correction off
def req2(train_ids, val, steps):
    rule(2, "Bias correction switched off")
    b1, b2 = 0.9, 0.999
    ratio = lambda t: (1 - b1 ** t) / math.sqrt(1 - b2 ** t)      # uncorrected step / corrected step
    say("  constant gradient: corrected Adam steps exactly eta every time; uncorrected steps")
    say("  eta * (1 - b1^t) / sqrt(1 - b2^t). The m bias shrinks the step, the v bias inflates it,")
    say("  and with b2 = 0.999 the v bias wins for thousands of steps.")
    hand_on = adam_by_hand([0.5] * 20, 1.0, 1e-3)
    hand_off = adam_by_hand([0.5] * 20, 1.0, 1e-3, bias_correction=False)
    first20 = [dict(t=a["t"], corrected=-a["step"] / 1e-3, uncorrected=-b["step"] / 1e-3) for a, b in zip(hand_on, hand_off)]
    say("")
    say("  step size / eta, first 20 steps:")
    for r in first20:
        if r["t"] in (1, 2, 3, 5, 10, 15, 20):
            say(f"    t={r['t']:>2}  corrected {r['corrected']:.3f}   uncorrected {r['uncorrected']:.3f}")
    peak_t = max(range(1, 200), key=ratio)
    cross = {thr: next(t for t in range(1, 100000) if t > peak_t and ratio(t) <= 1 + thr) for thr in (0.5, 0.1, 0.01)}
    say(f"  uncorrected peaks at {ratio(peak_t):.2f}x eta at t={peak_t}; comes within 50% at t={cross[0.5]}, "
        f"10% at t={cross[0.1]:,}, 1% at t={cross[0.01]:,}")
    say("  -> within twenty steps the difference never stops mattering; the uncorrected step is still 6x.")
    say("  The ratio is exact for ANY gradient sequence, not just a constant one: both versions divide the same")
    say("  m by the same sqrt(v), so step for step they differ by exactly (1 - b1^t) / sqrt(1 - b2^t) (eps aside).")

    # does it matter to a real model? Same model, same data; the custom optimizer, verified first.
    ref = train(train_ids, val, 1e-3, lambda s: 1.0, 20)
    mine = train(train_ids, val, 1e-3, lambda s: 1.0, 20, custom=True)
    worst = max(abs(a - b) for a, b in zip(ref["loss"], mine["loss"]))
    say("")
    say(f"  the switchable AdamW, checked first: 20 steps against torch.optim.AdamW, worst loss difference {worst:.1e}")
    runs = {}
    for warm, label in ((0, "no warmup"), (30, "30-step warmup")):
        sched = (lambda s, w=warm: min(1.0, (s + 1) / w)) if warm else (lambda s: 1.0)
        on = train(train_ids, val, 1e-3, sched, steps, custom=True, eval_at={steps})
        off = train(train_ids, val, 1e-3, sched, steps, custom=True, bias_correction=False, eval_at={steps})
        la, lb = _ema(on["loss"], 0.9), _ema(off["loss"], 0.9)
        rel = [abs(a - b) / a for a, b in zip(la, lb)]
        settle = next((s for s in range(len(rel)) if all(r < 0.01 for r in rel[s:])), None)
        runs[label] = dict(on_loss=on["loss"], off_loss=off["loss"], on_eval=on["evals"][steps],
                           off_eval=off["evals"][steps], settle_step=settle, max_rel_gap=max(rel))
        where = f"within 1% from step {settle} on" if settle is not None else "never back within 1% inside the run"
        say(f"  {label:<15} final held-out loss: corrected {on['evals'][steps]:.4f}, uncorrected {off['evals'][steps]:.4f};"
            f" training curves {where} (largest gap {100 * max(rel):.1f}%)")
        del on, off
    fig = plot_bias(first20, ratio, runs)
    return dict(first20=first20, peak_t=peak_t, peak_ratio=ratio(peak_t), crossings=cross,
                custom_vs_torch_max_loss_diff=worst, runs=runs, figure=fig)


# ============================================== 3. update-to-weight ratio
def smooth(xs, k=9):
    return [sorted(xs[max(0, i - k // 2):i + k // 2 + 1])[len(xs[max(0, i - k // 2):i + k // 2 + 1]) // 2]
            for i in range(len(xs))]


def req3(train_ids, val, steps, warm=100):
    rule(3, "The update-to-weight ratio, every layer, every step")
    say(f"  ||delta w|| / ||w|| per weight matrix, peak lr 1e-3, {steps} steps; linear warmup over {warm} steps")
    say(f"  against the same run with no warmup. Clipping at 1.0 in both.")
    with_w = train(train_ids, val, 1e-3, lambda s: min(1.0, (s + 1) / warm), steps, log_ratio=True)
    no_w = train(train_ids, val, 1e-3, lambda s: 1.0, steps, log_ratio=True)
    layers = list(with_w["ratios"])
    per = {}
    for n in layers:
        a, b = smooth(with_w["ratios"][n]), smooth(no_w["ratios"][n])
        rel = [abs(x - y) / y for x, y in zip(a, b)]
        stop = next((s for s in range(len(rel)) if all(r < 0.10 for r in rel[s:])), None)
        per[n] = dict(stop=stop, max_with=max(with_w["ratios"][n]), max_without=max(no_w["ratios"][n]),
                      final=sum(with_w["ratios"][n][-20:]) / 20)
    stops = sorted(v["stop"] for v in per.values() if v["stop"] is not None)
    # Per layer, the two runs' weights drift apart for reasons unrelated to warmup, so one layer's ratio
    # can differ forever. Across layers that washes out: take the median over layers of warmup / no-warmup
    # and find the step after which it stays within 10% of 1.
    sm_w = {n: smooth(with_w["ratios"][n]) for n in layers}
    sm_n = {n: smooth(no_w["ratios"][n]) for n in layers}
    q = [sorted(sm_w[n][s] / sm_n[n][s] for n in layers)[len(layers) // 2] for s in range(len(sm_w[layers[0]]))]
    med = next((s for s in range(len(q)) if all(abs(x - 1) < 0.10 for x in q[s:])), None)
    say("")
    say(f"  {'layer':<32} {'max, no warmup':>15} {'max, warmup':>12} {'last 20':>9} {'warmup stops mattering':>23}")
    for n, v in per.items():
        say(f"  {n:<32} {v['max_without']:>15.2e} {v['max_with']:>12.2e} {v['final']:>9.2e} "
            f"{('step ' + str(v['stop'])) if v['stop'] is not None else 'never (within run)':>23}")
    mw = max(v["max_with"] for v in per.values())
    mn = max(v["max_without"] for v in per.values())
    say("")
    say(f"  largest ratio anywhere: {mn:.2e} without warmup, {mw:.2e} with it ({mn / mw:.1f}x smaller)")
    say(f"  rule: the median over all {len(per)} layers of ratio(warmup) / ratio(no warmup), each smoothed with a")
    say(f"  9-step rolling median, stays within 10% of 1 from that step on.")
    say(f"  -> warmup stops changing the update-to-weight ratio at step {med}"
        f" (warmup itself ends at step {warm}); {len(stops)} of {len(per)} layers also pass the rule individually")
    fig = plot_ratio(with_w["ratios"], no_w["ratios"], med, warm)
    return dict(warmup_steps=warm, per_layer=per, median_stop=med, layers_passing=len(stops), median_ratio=q,
                max_with=mw, max_without=mn, ratios_with=with_w["ratios"], ratios_without=no_w["ratios"], figure=fig)


# ================================================== 4. cosine against WSD
def cosine(total, warm=30, floor=0.1):
    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def wsd(total, warm=30, decay_frac=0.2, floor=0.1):
    start = int(total * (1 - decay_frac))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        if s < start:
            return 1.0
        return 1.0 - (1 - floor) * min(1.0, (s - start) / max(1, total - start))
    return f


def req4(train_ids, val, quick):
    rule(4, "Cosine against WSD: 300-step schedules, both stopped at step 200")
    total, stop = 300, 200
    grid = [5e-4, 1e-3, 2e-3] if quick else [5e-4, 1e-3, 2e-3, 4e-3]
    say(f"  both schedules defined for {total} steps with a 30-step warmup; WSD decays over the last 20%")
    say(f"  (steps 240-300). Stopped at {stop}: cosine is at {cosine(total)(stop - 1):.2f}x its peak, WSD at "
        f"{wsd(total)(stop - 1):.2f}x -- it has not started to decay.")
    say(f"  tune both sides: peak learning rate swept over {grid} for each schedule first.")
    say("")
    say(f"  {'peak lr':>8} {'cosine @200':>12} {'WSD @200':>10}")
    tab = {}
    for lr in grid:
        c = train(train_ids, val, lr, cosine(total), total, stop_at=stop, eval_at={stop})
        w = train(train_ids, val, lr, wsd(total), total, stop_at=stop, eval_at={stop})
        tab[lr] = dict(cosine=c["evals"].get(stop, float("nan")), wsd=w["evals"].get(stop, float("nan")))
        say(f"  {lr:>8.0e} {tab[lr]['cosine']:>12.4f} {tab[lr]['wsd']:>10.4f}")
        del c, w
    best_c = min(grid, key=lambda lr: tab[lr]["cosine"])
    best_w = min(grid, key=lambda lr: tab[lr]["wsd"])
    say("")
    say(f"  tuned: cosine best at {best_c:.0e} -> {tab[best_c]['cosine']:.4f};  WSD best at {best_w:.0e} -> {tab[best_w]['wsd']:.4f}")

    # what WSD is actually for: branch a short decay off the step-200 checkpoint
    w = train(train_ids, val, best_w, wsd(total), total, stop_at=stop, eval_at={stop})
    k = 20
    branch = lambda s: 1.0 - 0.9 * min(1.0, (s - stop + 1) / k)
    wb = train(train_ids, val, best_w, branch, stop + k, model=w["model"], opt=w["opt"], start=stop,
               eval_at={stop + k})
    ref = train(train_ids, val, best_c, cosine(stop), stop, eval_at={stop})
    cur = dict(cosine_lr=[cosine(total)(s) * best_c for s in range(total)],
               wsd_lr=[wsd(total)(s) * best_w for s in range(total)])
    c_run = train(train_ids, val, best_c, cosine(total), total, stop_at=stop, eval_at=set(range(25, stop + 1, 25)))
    w_run = train(train_ids, val, best_w, wsd(total), total, stop_at=stop, eval_at=set(range(25, stop + 1, 25)))
    say(f"  WSD step-200 checkpoint + a {k}-step decay branch (10% more steps): {wb['evals'][stop + k]:.4f}")
    say(f"  reference: cosine scheduled for exactly {stop} steps (knowing the length in advance): {ref['evals'][stop]:.4f}")
    keep = "cosine" if tab[best_c]["cosine"] < tab[best_w]["wsd"] else "WSD"
    say("")
    say(f"  stopped at 200 with no further compute, the lower held-out loss is {keep}'s "
        f"({min(tab[best_c]['cosine'], tab[best_w]['wsd']):.4f} vs {max(tab[best_c]['cosine'], tab[best_w]['wsd']):.4f}).")
    fig = plot_sched(cur, c_run, w_run, stop)
    return dict(total=total, stop=stop, grid=grid, table={str(k_): v for k_, v in tab.items()},
                best_cosine_lr=best_c, best_wsd_lr=best_w, cosine_at_stop=tab[best_c]["cosine"],
                wsd_at_stop=tab[best_w]["wsd"], wsd_branch=wb["evals"][stop + k], branch_steps=k,
                cosine_200_reference=ref["evals"][stop], keep=keep,
                curves=dict(cosine=c_run["evals"], wsd=w_run["evals"]), figure=fig)


# ============================================= 5. learning rate across widths
def req5(train_ids, val, quick):
    rule(5, "Learning-rate sweep at widths 256, 512 and 1,024")
    widths = [256, 512, 1024]
    grid = [2e-4, 8e-4, 3.2e-3] if quick else [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3, 6.4e-3, 1.28e-2]
    steps = 120 if quick else 300
    say(f"  4 layers, head dim 64, standard parameterization (init std 0.02 at every width), batch {B}x{T},")
    say(f"  {steps} steps (30 warmup, cosine to 10%), one seed. Loss = held-out, token-weighted.")
    res = {}
    say("")
    say(f"  {'peak lr':>9} " + " ".join(f"{'w=' + str(w):>9}" for w in widths))
    table = {w: {} for w in widths}
    for w in widths:
        for lr in grid:
            torch.cuda.empty_cache()
            r = train(train_ids, val, lr, cosine(steps), steps, over=dict(d_model=w, n_head=w // 64),
                      eval_at={steps})
            table[w][lr] = r["evals"].get(steps, float("nan"))
            del r
    for lr in grid:
        say(f"  {lr:>9.1e} " + " ".join(f"{table[w][lr]:>9.4f}" for w in widths))
    say("")
    for w in widths:
        pts = [(lr, l) for lr, l in table[w].items() if math.isfinite(l)]
        i = min(range(len(pts)), key=lambda j: pts[j][1])
        grid_best = pts[i][0]
        # parabola through the minimum and its neighbours, in log2(lr)
        if 0 < i < len(pts) - 1:
            (x0, y0), (x1, y1), (x2, y2) = [(math.log2(pts[j][0]), pts[j][1]) for j in (i - 1, i, i + 1)]
            den = (x0 - x1) * (x0 - x2) * (x1 - x2)
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
            b_ = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
            fit = 2 ** (-b_ / (2 * a)) if a > 0 else grid_best
        else:
            fit = grid_best
        res[w] = dict(grid_best=grid_best, fitted=fit, loss=pts[i][1], edge=i in (0, len(pts) - 1))
        say(f"  width {w:>5}: grid minimum {grid_best:.1e} (loss {pts[i][1]:.4f}), parabola in log-lr {fit:.2e}"
            + ("  [minimum on the grid edge]" if res[w]["edge"] else ""))
    # power law lr* = a * width^k
    xs = [math.log(w) for w in widths]
    ys = [math.log(res[w]["fitted"]) for w in widths]
    mx, my = sum(xs) / 3, sum(ys) / 3
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    pred = math.exp(my + k * (math.log(4096) - mx))
    pair = [math.log(res[widths[i + 1]]["fitted"] / res[widths[i]]["fitted"]) / math.log(2) for i in range(2)]
    lo = min(res[1024]["fitted"] * 4 ** p for p in pair)
    hi = max(res[1024]["fitted"] * 4 ** p for p in pair)
    say("")
    say(f"  fitted lr* ~ width^{k:+.2f}. Extrapolated to width 4,096: {pred:.1e}")
    say(f"  the two width doublings alone give exponents {pair[0]:+.2f} and {pair[1]:+.2f} -> {lo:.1e} to {hi:.1e} at 4,096")
    fig = plot_sweep(table, res, widths, grid)
    return dict(widths=widths, grid=grid, steps=steps, table={str(w): {str(lr): v for lr, v in t.items()} for w, t in table.items()},
                minima={str(w): v for w, v in res.items()}, exponent=k, pair_exponents=pair,
                predicted_4096=pred, range_4096=[lo, hi], figure=fig)


# ================================================================ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def _save(fig, name):
    p = HERE / "figures" / name
    p.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    _plt().close(fig)
    return str(p.relative_to(HERE))


def _ema(xs, a=0.9):
    out, m = [], xs[0]
    for x in xs:
        m = a * m + (1 - a) * x
        out.append(m)
    return out


def plot_bias(first20, ratio, runs):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), facecolor=SURFACE)
    t = [r["t"] for r in first20]
    a1.plot(t, [r["uncorrected"] for r in first20], color=ORANGE, lw=1.6, marker="o", ms=4, label="bias correction off")
    a1.plot(t, [r["corrected"] for r in first20], color=BLUE, lw=1.6, marker="o", ms=4, label="bias correction on")
    a1.text(t[-1], first20[-1]["uncorrected"], "  off", color=INK2, va="center", fontsize=9)
    a1.text(t[-1], first20[-1]["corrected"], "  on", color=INK2, va="center", fontsize=9)
    _style(a1, "Step size in units of eta, constant gradient, first 20 steps", "step / eta", "step")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK2)
    r = runs["no warmup"]
    s = list(range(len(r["on_loss"])))
    a2.plot(s, _ema(r["off_loss"]), color=ORANGE, lw=1.6, label="bias correction off")
    a2.plot(s, _ema(r["on_loss"]), color=BLUE, lw=1.6, label="bias correction on")
    _style(a2, "A real model, no warmup: training loss (EMA 0.9)", "loss", "step")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK2)
    return _save(fig, "bias_correction.png")


def plot_ratio(rw, rn, med, warm):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    n = len(next(iter(rw.values())))
    xs = list(range(n))

    def band(r):
        cols = list(zip(*r.values()))
        return [sorted(c)[len(c) // 2] for c in cols], [min(c) for c in cols], [max(c) for c in cols]
    for r, col, lab in ((rn, ORANGE, "no warmup"), (rw, BLUE, f"{warm}-step warmup")):
        med_, lo, hi = band(r)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(xs, med_, color=col, lw=1.6, label=f"{lab} (median layer; band = min-max)")
    ax.axhline(1e-3, color=INK2, lw=1, ls=":")
    ax.text(n - 1, 1e-3, "1e-3 target ", color=INK2, fontsize=8, ha="right", va="bottom")
    if med is not None:
        ax.axvline(med, color=INK2, lw=1, ls="--")
        ax.text(med, max(max(v) for v in rn.values()), f"step {med}: warmup stops mattering ",
                color=INK, fontsize=9, va="top", ha="right")
    ax.set_yscale("log")
    _style(ax, "Update-to-weight ratio across all weight matrices", "||delta w|| / ||w||", "step")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    return _save(fig, "update_to_weight.png")


def plot_sched(cur, c_run, w_run, stop):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), facecolor=SURFACE)
    s = list(range(len(cur["cosine_lr"])))
    a1.plot(s, cur["cosine_lr"], color=BLUE, lw=1.6, label="cosine")
    a1.plot(s, cur["wsd_lr"], color=ORANGE, lw=1.6, label="WSD")
    a1.axvline(stop, color=INK2, lw=1, ls="--")
    a1.text(stop, max(cur["wsd_lr"]), f" stop at {stop}", color=INK, fontsize=9, va="top")
    _style(a1, "Learning rate, 300-step schedules at each one's tuned peak", "learning rate")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK2)
    for run, col, lab in ((c_run, BLUE, "cosine"), (w_run, ORANGE, "WSD")):
        e = sorted(run["evals"].items())
        a2.plot([k for k, _ in e], [v for _, v in e], color=col, lw=1.6, marker="o", ms=4, label=lab)
        a2.text(e[-1][0], e[-1][1], f"  {lab} {e[-1][1]:.3f}", color=INK2, fontsize=9,
                va="bottom" if lab == "cosine" else "top")
    _style(a2, "Held-out loss up to the stop", "held-out loss", "step")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK2)
    return _save(fig, "cosine_vs_wsd.png")


def plot_sweep(table, res, widths, grid):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    cols = {256: BLUE, 512: ORANGE, 1024: AQUA}
    finite = [v for w in widths for v in table[w].values() if math.isfinite(v)]
    top = sorted(finite)[int(0.8 * len(finite))] + 0.3
    for w in widths:
        pts = [(lr, l) for lr, l in table[w].items() if math.isfinite(l) and l < top]
        m = res[w]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cols[w], lw=1.6, marker="o", ms=5,
                label=f"width {w}: minimum at {m['fitted']:.1e}")
        ax.plot([m["grid_best"]], [m["loss"]], marker="*", ms=16, color=cols[w], mec=SURFACE, mew=1.2)
        ax.text(pts[0][0] * 0.93, pts[0][1], f"width {w}", color=INK, fontsize=9, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(min(grid) * 0.45, max(grid) * 1.3)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center")
    _style(ax, "Held-out loss against peak learning rate (star = each width's minimum)", "held-out loss", "peak learning rate")
    return _save(fig, "lr_sweep.png")


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw figures/ from results.json, no training")
    args = ap.parse_args()
    if args.replot:
        R = json.loads((HERE / "results.json").read_text())
        r2, r3, r4, r5 = R["req2"], R["req3"], R["req4"], R["req5"]
        b1, b2 = 0.9, 0.999
        plot_bias(r2["first20"], lambda t: (1 - b1 ** t) / math.sqrt(1 - b2 ** t), r2["runs"])
        plot_ratio(r3["ratios_with"], r3["ratios_without"], r3["median_stop"], r3["warmup_steps"])
        cur = dict(cosine_lr=[cosine(300)(k) * r4["best_cosine_lr"] for k in range(300)],
                   wsd_lr=[wsd(300)(k) * r4["best_wsd_lr"] for k in range(300)])
        ev = lambda d: dict(evals={int(k): v for k, v in d.items()})
        plot_sched(cur, ev(r4["curves"]["cosine"]), ev(r4["curves"]["wsd"]), r4["stop"])
        widths = r5["widths"]
        table = {w: {float(k): v for k, v in r5["table"][str(w)].items()} for w in widths}
        plot_sweep(table, {w: r5["minima"][str(w)] for w in widths}, widths, r5["grid"])
        print("figures redrawn")
        return
    steps = 120 if args.quick else 300
    t0 = time.time()
    say(f"ERA V5 Session 11 -- optimizers | seed {SEED} | device {DEV}"
        + (f" ({torch.cuda.get_device_name()})" if DEV == "cuda" else "") + f" | torch {torch.__version__}")
    train_ids, val = token_splits()
    say(f"tiny Shakespeare, GPT-2 BPE: {len(train_ids):,} train / {len(val):,} val tokens; batch {B}x{T}")
    R = {}
    R["req1"] = req1()
    R["req2"] = req2(train_ids, val, steps)
    R["req3"] = req3(train_ids, val, steps)
    R["req4"] = req4(train_ids, val, args.quick)
    R["req5"] = req5(train_ids, val, args.quick)
    R["meta"] = dict(seed=SEED, torch=torch.__version__, device=DEV,
                     gpu=torch.cuda.get_device_name() if DEV == "cuda" else None,
                     quick=args.quick, config=CFG, batch=[B, T], seconds=round(time.time() - t0))
    say("")
    say(f"done in {time.time() - t0:.0f}s")
    (HERE / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (HERE / "run.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
