"""Session 14 assignment: train a dense model, grow it into a mixture of experts, keep training.

    python train.py --quick     # a few minutes, tiny token budgets, proves the plumbing
    python train.py             # the real run; writes run.log, results.json

The assignment asks for two things: that the converted model continues to train, and that the loss
keeps dropping. Showing that needs a dense baseline that is actually trained, so the run is in
stages and each stage saves results.json before the next one starts.

    1 dense        train the dense model for 12M tokens; this is what gets grown
    2 exactness    force the router onto the inherited experts and check the MoE layer reproduces
                   the dense layer to float precision
    3 convert      for each growth method, the validation loss the instant the model is converted,
                   with and without the down-projection scaling
    4 grow         continue training each grown model for 8M tokens, against a dense control that
                   gets the same 8M tokens
    5 balance      the same grown model under no balancing, Switch's auxiliary loss, and the
                   auxiliary-loss-free bias; MaxVio and dead experts for each
    6 routing      the clone init under hard top-k against probabilistic selection, which is the
                   fix the V4 cookbook reports for collapse along clone families
    7 scope        the bias window counted per micro-batch against per whole batch

Every number in README.md comes from results.json, which comes from this script. Nothing is typed
in by hand.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import data
import model as M

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20261010
BLOCK = 512
LR = 3e-4
WARMUP = 100
R = {}


def log(*a):
    print(*a, flush=True)


def save():
    (HERE / "results.json").write_text(json.dumps(R, indent=1))


# ------------------------------------------------------------------------------- data


class Batches:
    def __init__(self, split, batch):
        self.d = np.asarray(data.load(split))
        self.batch = batch
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(SEED)

    def __call__(self):
        ix = self.rng.integers(0, len(self.d) - BLOCK - 1, self.batch)
        x = np.stack([self.d[i:i + BLOCK] for i in ix])
        y = np.stack([self.d[i + 1:i + 1 + BLOCK] for i in ix])
        t = (torch.from_numpy(x).long().to(DEV, non_blocking=True),
             torch.from_numpy(y).long().to(DEV, non_blocking=True))
        return t


@torch.no_grad()
def evaluate(net, batches, iters=40):
    """Always the same 40 batches, so two models' losses can be compared directly."""
    batches.reset()
    net.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = batches()
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            _, loss = net(x, y)
        tot += loss.item()
    net.train()
    return tot / iters


# ------------------------------------------------------------------------------- training


def run(net, tag, tokens, batch, accum=1, lr=LR, mon_last=0.2):
    """Train `net` for `tokens` tokens. Returns a result row; logs every 50 steps."""
    net.to(DEV).train()
    tr, va = Batches("train", batch), Batches("val", batch)
    cfg = net.cfg
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    per_step = batch * BLOCK * accum
    steps = max(1, tokens // per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 *
                                                     (1 + math.cos(math.pi * min(1.0, s / steps)))))
    mon_from = int(steps * (1 - mon_last))
    if cfg.moe:
        net.reset_mon()
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    curve, t0, prob_left = [], time.time(), cfg.prob_routing_steps
    for s in range(steps):
        if cfg.moe and prob_left and s == prob_left:
            cfg.prob_routing_steps = 0          # the window is over; hard top-k from here
            log(f"[{tag}] probabilistic selection off at step {s}")
        if cfg.moe and s == mon_from:
            net.reset_mon()                     # measure the load over the last stretch only
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            x, y = tr()
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                _, loss = net(x, y)
                if cfg.moe and cfg.balance == "aux":
                    aux = net.aux_loss()
                    if aux is not None:
                        loss = loss + aux / cfg.n_layer
            (loss / accum).backward()
            tot += loss.item() / accum
            if cfg.moe and cfg.balance == "bias" and cfg.scope == "micro":
                net.step_bias(cfg.bias_gamma)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if cfg.moe and cfg.balance == "bias" and cfg.scope == "batch":
            net.step_bias(cfg.bias_gamma)
        curve.append(round(tot, 4))
        if s % 50 == 0 or s == steps - 1:
            log(f"[{tag}] step {s}/{steps} loss {tot:.4f} lr {sched.get_last_lr()[0]:.2e}")
    secs = time.time() - t0
    val = evaluate(net, va)
    row = dict(tag=tag, tokens=steps * per_step, steps=steps, batch=batch, accum=accum, lr=lr,
               val_loss=round(val, 4), final_train_loss=round(float(np.mean(curve[-20:])), 4),
               seconds=round(secs, 1), tokens_per_s=round(steps * per_step / secs),
               peak_mem_mib=round(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None,
               curve=curve[::max(1, steps // 60)])
    if cfg.moe:
        st = net.load_stats()
        row.update(maxvio=round(st["maxvio"], 4), maxvio_worst=round(st["maxvio_worst"], 4),
                   dead=st["dead"], dead_of=st["dead_total"],
                   balance=cfg.balance, scope=cfg.scope,
                   layer0_share=[round(v, 5) for v in st["per_layer_share"][0]])
    log(f"[{tag}] done: val {row['val_loss']} · {row['tokens_per_s']:,} tok/s · "
        f"{row['peak_mem_mib']:,} MiB peak" +
        (f" · MaxVio {row['maxvio']} · {row['dead']}/{row['dead_of']} dead" if cfg.moe else ""))
    return row


# ------------------------------------------------------------------------------- stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    q = args.quick
    dense_tokens = 400_000 if q else 12_000_000
    grow_tokens = 200_000 if q else 8_000_000
    batch, accum = (4, 1) if q else (8, 2)

    R.update(device=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
             torch=torch.__version__, seed=SEED, block=BLOCK, quick=q,
             dense_tokens=dense_tokens, grow_tokens=grow_tokens, batch=batch, accum=accum)

    dcfg = M.Config()
    mcfg = lambda **kw: M.Config(moe=True, **kw)
    dense = M.GPT(dcfg)
    R["shape"] = dict(
        n_layer=dcfg.n_layer, d_model=dcfg.d_model, n_head=dcfg.n_head, block=BLOCK,
        dense_inner=M.dense_inner(dcfg.d_model), n_expert=mcfg().n_expert,
        top_k=mcfg().top_k, expert_inner=mcfg().expert_inner,
        dense_params=dense.param_counts(), moe_params=M.GPT(mcfg()).param_counts())
    log(f"device {DEV} · {R['device']} · torch {torch.__version__} · seed {SEED}")
    log(f"dense {R['shape']['dense_params']['total']:,} params · "
        f"moe {R['shape']['moe_params']['total']:,} total / "
        f"{R['shape']['moe_params']['active']:,} active")
    save()

    # ---- 1. the dense model
    log("\n== 1. dense model")
    R["dense"] = run(dense, "dense", dense_tokens, batch, accum)
    dense_val = R["dense"]["val_loss"]
    dense = dense.cpu()
    torch.save(dense.state_dict(), HERE / "dense.pt")
    save()

    # ---- 2. exactness of the conversion
    log("\n== 2. conversion exactness (router forced onto the inherited experts)")
    va = Batches("val", batch)
    R["exactness"] = []
    for meth in ("clone", "partition", "drop"):
        net = M.upcycle(dense, meth, mcfg(balance="bias"), scale_down=True).to(DEV)
        for lay in net.moe_layers():
            torch.nn.init.zeros_(lay.router.weight)
            lay.bias.zero_()
            lay.bias[:mcfg().top_k] = 1.0        # choosing bias only; weights stay uniform
        v = evaluate(net, va)
        R["exactness"].append(dict(method=meth, val_loss=round(v, 6),
                                   delta=round(v - dense_val, 6)))
        log(f"  {meth:10s} val {v:.6f}  delta {v - dense_val:+.6f}")
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()
    save()

    # ---- 3. the loss at the moment of conversion
    log("\n== 3. validation loss at the moment of conversion")
    R["convert"] = []
    for meth in ("random", "clone", "partition", "drop"):
        for sc in (False, True):
            net = M.upcycle(dense, meth, mcfg(), scale_down=sc).to(DEV)
            v = evaluate(net, va)
            R["convert"].append(dict(method=meth, scale_down=sc, val_loss=round(v, 4),
                                     delta=round(v - dense_val, 4)))
            log(f"  {meth:10s} scale_down={sc!s:5s} val {v:.4f}  delta {v - dense_val:+.4f}")
            del net
            if DEV == "cuda":
                torch.cuda.empty_cache()
    save()
    best_scale = {m: min((r for r in R["convert"] if r["method"] == m),
                         key=lambda r: r["val_loss"])["scale_down"]
                  for m in ("random", "clone", "partition", "drop")}
    R["best_scale"] = best_scale
    log(f"  carrying forward scale_down = {best_scale}")

    # ---- 4. continue training
    log("\n== 4. continued training, grown models against a dense control")
    R["grow"] = []
    ctrl = M.GPT(dcfg)
    ctrl.load_state_dict(torch.load(HERE / "dense.pt"))
    R["grow"].append(run(ctrl, "dense-control", grow_tokens, batch, accum))
    del ctrl
    if DEV == "cuda":
        torch.cuda.empty_cache()
    save()
    for meth in ("random", "clone", "partition", "drop"):
        net = M.upcycle(dense, meth, mcfg(balance="bias"), scale_down=best_scale[meth])
        row = run(net, f"grow:{meth}", grow_tokens, batch, accum)
        row.update(method=meth, scale_down=best_scale[meth])
        R["grow"].append(row)
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()
        save()

    # ---- 5. balancing
    log("\n== 5. balancing methods, on the partition init")
    R["balance"] = [dict(r) for r in R["grow"] if r.get("method") == "partition"]
    for bal in ("none", "aux"):
        net = M.upcycle(dense, "partition", mcfg(balance=bal), scale_down=best_scale["partition"])
        row = run(net, f"balance:{bal}", grow_tokens, batch, accum)
        row.update(method="partition")
        R["balance"].append(row)
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()
        save()

    # ---- 6. routing during the growth window
    #
    # Both arms run with balancing switched off. With the bias on, the bias is what keeps the
    # clones alive and the routing rule cannot be seen; the V4 cookbook's failure was measured
    # under hard top-k, so that is the condition to reproduce.
    log("\n== 6. hard top-k against probabilistic selection, on the clone init, no balancing")
    R["routing"] = []
    prob_steps = 50 if q else 200
    for name, ps in (("hard", 0), ("prob", prob_steps)):
        net = M.upcycle(dense, "clone", mcfg(balance="none", prob_routing_steps=ps),
                        scale_down=best_scale["clone"])
        row = run(net, f"routing:{name}", grow_tokens, batch, accum)
        row.update(method="clone",
                   routing="hard top-k" if ps == 0 else f"probabilistic for {ps} steps")
        R["routing"].append(row)
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()
        save()

    # ---- 7. balancing scope
    log("\n== 7. the bias window: per micro-batch against per whole batch")
    R["scope"] = [dict(r) for r in R["grow"] if r.get("method") == "partition"]
    net = M.upcycle(dense, "partition", mcfg(balance="bias", scope="micro"),
                    scale_down=best_scale["partition"])
    row = run(net, "scope:micro", grow_tokens, batch, accum)
    row.update(method="partition")
    R["scope"].append(row)
    save()
    log("\nwrote results.json")


if __name__ == "__main__":
    main()
