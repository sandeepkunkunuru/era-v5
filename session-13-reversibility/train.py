"""Session 13 — reversibility, measured.

The assignment, as given in class [01:30:57], is three runs:

    1. baseline at the largest batch that fits         -- speed, memory, final loss
    2. reversible at the SAME batch                    -- and which variant works
    3. reversible at the largest batch IT allows       -- does the recovered memory buy throughput?

plus, from the Q&A [01:38:41], the cost of each. Four things are measured beyond the brief:

    4. reconstruction error   -- the property the whole method rests on, per variant and per
                                 damping coefficient a (V4 shipped a = 0.5; the paper requires
                                 |a| = 1 for forward-backward stability)
    5. memory against depth   -- the paper's headline claim is that reversible activation memory is
                                 constant in depth while the baseline's is linear
    6. gradient equivalence   -- the custom backward must agree with ordinary autograd
    7. dropout                -- why the baseline must also run at dropout 0 for the comparison to
                                 be between two stacks rather than two models

    python train.py            # everything: ~4 h on one RTX 4050
    python train.py --quick    # short runs, for checking the wiring
"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

# Reversible backward allocates and frees a block's worth of activations at every layer, which
# fragments the allocator badly on a small card. Expandable segments keep those reuses contiguous.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from data import load
from model import Config, GPT

HERE = Path(__file__).resolve().parent
SEED = 20260919                       # the session date
TOKENS = 50_000_000
BLOCK = 512
LR, BETAS, WD, CLIP = 3e-4, (0.9, 0.95), 0.0, 1.0   # weight decay 0: see README
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def batches(data, batch, block, steps, seed):
    g = np.random.default_rng(seed)
    for _ in range(steps):
        ix = g.integers(0, len(data) - block - 1, size=batch)
        x = np.stack([data[i:i + block] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


def make(stack, **kw):
    cfg = Config(stack=stack, block_size=BLOCK, **kw)
    torch.manual_seed(SEED)
    return GPT(cfg).to(DEV), cfg


def _free():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def fits(stack, batch, layers=12, steps=3, **kw):
    """Can this batch survive a few *real* steps -- same code path as run(), clipping included?

    Two steps of forward+backward is not enough of a test: the optimizer state appears on step 1,
    and the allocator only settles after a few reuses. The quick run found a batch that passed a
    two-step probe and then ran out of memory in training, so the probe now mirrors the real loop.
    """
    _free()
    ok = True
    m = opt = x = loss = None
    try:
        m, _ = make(stack, n_layer=layers, **kw)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
        x = torch.randint(0, 256, (batch, BLOCK), device=DEV)
        for _ in range(steps):
            _, loss = m(x, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError:
        ok = False
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        ok = False
    finally:
        del m, opt, x, loss
        _free()
    return ok


def max_batch(stack, lo=1, hi=256, **kw):
    """Largest batch that survives a few real steps. Doubling, then bisection."""
    best = 0
    b = lo
    while b <= hi and fits(stack, b, **kw):
        best, b = b, b * 2
    lo, hi = best, min(b, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(stack, mid, **kw):
            lo = mid
        else:
            hi = mid
        best = lo
    return max(best, 1)


def run(stack, batch, tokens, tag, **kw):
    """One training run, retrying at a smaller batch if the card cannot hold this one.

    Returns loss curve, throughput, peak memory -- and records any back-off, because a batch that
    passes the probe and fails in training is itself a finding about how tight the card is.
    """
    for attempt in range(4):
        try:
            return _run_once(stack, batch, tokens, tag, **kw)
        except torch.cuda.OutOfMemoryError:
            _free()
            batch = max(1, int(batch * 0.85))
            print(f"  [{tag}] out of memory -- backing off to batch {batch}", flush=True)
    raise RuntimeError(f"{tag}: could not find a batch that fits")


def _run_once(stack, batch, tokens, tag, **kw):
    data, val = load("train"), load("val")
    m, cfg = make(stack, **kw)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
    steps = max(1, tokens // (batch * BLOCK))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.02, anneal_strategy="cos")
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, t0, seen = [], time.perf_counter(), 0
    for i, (x, y) in enumerate(batches(data, batch, BLOCK, steps, SEED)):
        _, loss = m(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        seen += x.numel()
        if i % 25 == 0 or i == steps - 1:
            losses.append((i, round(loss.item(), 4)))
        if i % 200 == 0:
            print(f"  [{tag}] step {i}/{steps} loss {loss.item():.4f}", flush=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    m.eval()
    with torch.no_grad():
        vl = [m(x, y)[1].item() for x, y in batches(val, min(batch, 8), BLOCK, 20, 7)]
    m.train()
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0.0
    out = dict(tag=tag, stack=stack, batch=batch, steps=steps, tokens=seen,
               params=m.n_params(), h=cfg.h, a=cfg.a, layers=cfg.n_layer,
               final_train_loss=round(losses[-1][1], 4), val_loss=round(sum(vl) / len(vl), 4),
               seconds=round(dt, 1), tokens_per_s=round(seen / dt, 1),
               peak_mem_mib=round(peak, 1), curve=losses)
    del m, opt
    _free()
    print(f"  [{tag}] done: val {out['val_loss']:.4f} · {out['tokens_per_s']:,.0f} tok/s · "
          f"{out['peak_mem_mib']:,.0f} MiB peak", flush=True)
    return out


def reconstruction_table():
    """The property the method rests on: does the rebuilt state match the real one?"""
    rows = []
    x = torch.randint(0, 256, (2, BLOCK), device=DEV)
    for stack in ("midpoint", "leapfrog", "hamiltonian"):
        for a in (1.0, 0.5):
            m, _ = make(stack, a=a)
            m = m.double()
            with torch.no_grad():
                err = m.reconstruction_error(x)
            rows.append(dict(stack=stack, a=a, rel_error=err))
            del m
            _free()
    return rows


def grad_check():
    """The custom backward against ordinary autograd, in fp64."""
    import torch.nn.functional as F
    rows = []
    for stack in ("midpoint", "leapfrog", "hamiltonian"):
        cfg = Config(n_layer=4, d_model=64, n_head=4, block_size=32, stack=stack)
        torch.manual_seed(SEED)
        m = GPT(cfg).to(DEV).double()
        x = torch.randint(0, 256, (2, 32), device=DEV)
        m.zero_grad()
        _, loss = m(x, x)
        loss.backward()
        g = {n: p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad()
        p = m.tok(x) + m.pos(torch.arange(32, device=DEV))
        if stack == "midpoint":
            prev, cur = p, p + cfg.h * m.blocks[0](p)
            for blk in m.blocks[1:]:
                prev, cur = cur, cfg.a * prev + 2 * cfg.h * blk(cur)
        elif stack == "leapfrog":
            prev, cur = p, p + 0.5 * cfg.h ** 2 * m.blocks[0](p)
            for blk in m.blocks[1:]:
                prev, cur = cur, 2 * cur - prev + cfg.h ** 2 * blk(cur)
        else:
            q, cur = torch.zeros_like(p), p
            for blk in m.blocks:
                q = cfg.a * q + blk.attn_part(cur)
                cur = cfg.a * cur + blk.mlp_part(q)
        logits = m.head(m.ln_f(cur))
        F.cross_entropy(logits.view(-1, logits.size(-1)), x.reshape(-1)).backward()
        worst = max(((p.grad - g[n]).norm() / g[n].norm().clamp_min(1e-30)).item()
                    for n, p in m.named_parameters() if p.grad is not None and g[n].norm() > 0)
        rows.append(dict(stack=stack, worst_rel_grad_error=worst))
        del m
        _free()
    return rows


def memory_vs_depth(batch, depths):
    """The paper's Figure 3 claim: linear in depth for the baseline, constant for reversible."""
    rows = []
    for L in depths:
        for stack in ("baseline", "midpoint"):
            _free()
            m = opt = x = loss = None
            peak, params, oom = 0.0, 0, False
            try:
                m, _ = make(stack, n_layer=L)
                params = m.n_params()
                opt = torch.optim.AdamW(m.parameters(), lr=LR)
                x = torch.randint(0, 256, (batch, BLOCK), device=DEV)
                _, loss = m(x, x)
                loss.backward()
                opt.step()
                peak = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0.0
            except torch.cuda.OutOfMemoryError:
                oom = True                       # a depth this stack cannot reach is itself the result
            finally:
                del m, opt, x, loss
                _free()
            rows.append(dict(layers=L, stack=stack, peak_mem_mib=round(peak, 1),
                             params=params, oom=oom))
            print(f"  {L:3d} layers  {stack:11s} "
                  + ("out of memory" if oom else f"{peak:8,.0f} MiB"), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    tokens = 2_000_000 if a.quick else TOKENS
    variant_tokens = 500_000 if a.quick else 4_000_000

    print(f"device {DEV} · {torch.cuda.get_device_name(0) if DEV == 'cuda' else ''} · "
          f"torch {torch.__version__} · seed {SEED}", flush=True)

    def save():
        """Write after every stage. The first version of this script lost three hours of finished
        runs when the depth sweep ran out of memory before anything had been written."""
        (HERE / "results.json").write_text(json.dumps(R, indent=1))

    R = dict(device=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
             torch=torch.__version__, seed=SEED, block=BLOCK, tokens=tokens, quick=a.quick)

    print("\n== correctness: custom backward vs autograd (fp64)", flush=True)
    R["grad_check"] = grad_check()
    save()
    for r in R["grad_check"]:
        print(f"  {r['stack']:12s} worst relative gradient error {r['worst_rel_grad_error']:.2e}")

    print("\n== reconstruction error, and what V4's a = 0.5 does to it", flush=True)
    R["reconstruction"] = reconstruction_table()
    save()
    for r in R["reconstruction"]:
        print(f"  {r['stack']:12s} a={r['a']}  relative error {r['rel_error']:.3e}")

    print("\n== largest batch that fits", flush=True)
    R["max_batch"] = {s: max_batch(s) for s in ("baseline", "midpoint", "leapfrog", "hamiltonian")}
    save()
    for s, b in R["max_batch"].items():
        print(f"  {s:12s} {b}")

    base_batch = R["max_batch"]["baseline"]

    print("\n== which reversible variant works (short runs, equal batch)", flush=True)
    R["variants"] = [run(s, base_batch, variant_tokens, f"variant:{s}")
                     for s in ("midpoint", "leapfrog", "hamiltonian")]
    best = min(R["variants"], key=lambda r: r["val_loss"])["stack"]
    R["best_variant"] = best
    save()
    print(f"  best by validation loss: {best}", flush=True)

    print("\n== the three runs the assignment asks for", flush=True)
    R["runs"] = []
    for stack, batch, tag in ((("baseline"), base_batch, "1-baseline"),
                              (best, base_batch, "2-reversible-same-batch"),
                              (best, R["max_batch"][best], "3-reversible-max-batch")):
        R["runs"].append(run(stack, batch, tokens, tag))
        save()

    print("\n== memory against depth", flush=True)
    R["memory_vs_depth"] = memory_vs_depth(max(1, base_batch // 2),
                                           [4, 8, 12] if a.quick else [4, 8, 12, 24, 48])

    save()
    print(f"\nwrote results.json", flush=True)


if __name__ == "__main__":
    main()
