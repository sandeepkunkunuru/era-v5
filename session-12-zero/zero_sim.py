"""ERA V5 · Session 12 — 32 virtual GPUs, and ZeRO stages 0 to 3 written out by hand.

Each "GPU" is a CPU process; the 32 of them talk through torch.distributed (gloo), using the same
three collectives a real run uses: all-reduce, reduce-scatter and all-gather. No DeepSpeed and no FSDP:
every stage below is the collectives called in the right order, so the mechanism is on the page.

    stage 0  data parallelism: everyone holds everything; all-reduce the gradients
    stage 1  shard the optimizer state (fp32 master copy + Adam's m and v)
    stage 2  ... and the gradients: reduce-scatter each one as soon as backward produces it, then drop it
    stage 3  ... and the weights: gather a layer's weights just before it runs, free them straight after

What is measured, per virtual GPU (rank), per stage:
    memory   the bytes of every persistent tensor the rank actually holds, by category, plus the largest
             transient (a gathered layer, a full gradient) -- counted from real tensors, not a formula
    traffic  every collective's payload, in units of P = the size of the whole model in bf16
    time     forward/backward compute, time inside collectives, and the optimizer step
    loss     the loss curve -- which must be the same in every stage, and the same as one process
             training on the whole global batch, or the sharding changed the maths

    python zero_sim.py              # full: stages 0-3 at world sizes 8, 16 and 32
    python zero_sim.py --quick      # world size 8 only, fewer steps
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SEED = 20260912                                  # the session date
CFG = dict(d=128, layers=2, heads=4, T=64, B=4)  # B = sequences per rank per step
LR, BETAS, EPS = 1e-3, (0.9, 0.999), 1e-8
FIGDIR = HERE / "figures"

# Palette validated with the dataviz skill's checker; aqua and yellow sit below 3:1 on the surface,
# so every mark carrying them is labelled and the numbers are tabled in the README.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


# ============================================================ data (char level)
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text():
    p = HERE / "input.txt"
    if not p.exists():
        urllib.request.urlretrieve(SHAKESPEARE, p)
    return p.read_text()


def encode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return torch.tensor([stoi[c] for c in text], dtype=torch.long), len(chars)


def global_batch(ids, step, n_seq, T):
    """The whole step's batch, identical on every rank. Rank r trains on slice r of it."""
    g = torch.Generator().manual_seed(SEED * 1000 + step)
    i = torch.randint(len(ids) - T - 1, (n_seq,), generator=g)
    seq = torch.stack([ids[j:j + T + 1] for j in i])
    return seq[:, :T], seq[:, 1:]


# ============================================================ the model's parameters
def param_shapes(V, d, layers):
    """A tiny GPT in which every parameter is a matrix: embeddings, attention and MLP projections, head.
    RMSNorm is written without a gain, so there is nothing else to shard."""
    s = {"emb": (V, d), "pos": (CFG["T"], d)}
    for l in range(layers):
        s[f"l{l}.qkv"] = (3 * d, d)
        s[f"l{l}.proj"] = (d, d)
        s[f"l{l}.fc1"] = (4 * d, d)
        s[f"l{l}.fc2"] = (d, 4 * d)
    s["head"] = (V, d)
    return s


def init_params(shapes):
    g = torch.Generator().manual_seed(SEED)
    return {k: torch.randn(*v, generator=g) * 0.02 for k, v in shapes.items()}   # fp32, identical everywhere


# ============================================================ the ledgers
class Ledger:
    """Counts every collective: logical bytes each rank puts on the wire, and time spent inside.

    Volumes follow the ring algorithms the brief describes. For a tensor of S bytes on W ranks:
    all-reduce moves 2(W-1)/W * S per rank; reduce-scatter and all-gather move (W-1)/W * S each."""

    def __init__(self, W):
        self.W = W
        self.bytes = {"all_reduce": 0.0, "reduce_scatter": 0.0, "all_gather": 0.0}
        self.calls = {k: 0 for k in self.bytes}
        self.seconds = 0.0
        self.on = True

    def _count(self, op, full_bytes, t0):
        self.seconds += time.perf_counter() - t0          # time: every step
        if self.on:                                        # volume: exactly one step
            f = 2 * (self.W - 1) / self.W if op == "all_reduce" else (self.W - 1) / self.W
            self.bytes[op] += f * full_bytes
            self.calls[op] += 1

    def all_reduce(self, t):
        t0 = time.perf_counter()
        dist.all_reduce(t)
        self._count("all_reduce", t.numel() * t.element_size(), t0)

    def reduce_scatter(self, out, full):
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out, full)
        self._count("reduce_scatter", full.numel() * full.element_size(), t0)

    def all_gather(self, full, shard):
        t0 = time.perf_counter()
        dist.all_gather_into_tensor(full, shard)
        self._count("all_gather", full.numel() * full.element_size(), t0)


class Transient:
    """The largest short-lived allocation a stage makes: a gathered weight, a full gradient."""

    def __init__(self):
        self.now = 0
        self.peak = 0

    def add(self, n):
        self.now += n
        self.peak = max(self.peak, self.now)

    def sub(self, n):
        self.now -= n


LEDGER, TRANSIENT = None, None


def chunk(numel, W):
    return math.ceil(numel / W)


def padded(t, W):
    """Flatten and pad to a multiple of W, so every rank's shard is the same size."""
    f = t.reshape(-1)
    return F.pad(f, (0, chunk(f.numel(), W) * W - f.numel()))


# ============================================================ stage-3 autograd functions
class ZLinear(torch.autograd.Function):
    """y = x W^T, where this rank holds only a shard of W.

    forward : all-gather W, compute, free W. Only the shard is saved for backward -- if the full W were
              saved (which is what autograd does for an ordinary matmul) every layer's weights would stay
              alive until backward and stage 3 would save nothing.
    backward: all-gather W again, compute dL/dx and dL/dW, reduce-scatter dL/dW straight back into shards."""

    @staticmethod
    def forward(ctx, x, shard, shape):
        W = dist.get_world_size()
        full = torch.empty(chunk(math.prod(shape), W) * W, dtype=shard.dtype)
        LEDGER.all_gather(full, shard)
        TRANSIENT.add(full.numel() * full.element_size())
        w = full[:math.prod(shape)].view(shape)
        y = x @ w.t()
        ctx.save_for_backward(x, shard)
        ctx.shape = shape
        TRANSIENT.sub(full.numel() * full.element_size())
        del full, w
        return y

    @staticmethod
    def backward(ctx, gy):
        x, shard = ctx.saved_tensors
        shape = ctx.shape
        W = dist.get_world_size()
        full = torch.empty(chunk(math.prod(shape), W) * W, dtype=shard.dtype)
        LEDGER.all_gather(full, shard)
        TRANSIENT.add(full.numel() * full.element_size())
        w = full[:math.prod(shape)].view(shape)
        gx = gy @ w
        gw = gy.reshape(-1, shape[0]).t().float() @ x.reshape(-1, shape[1]).float()
        g_shard = torch.empty_like(shard)
        gw_p = padded(gw.to(shard.dtype), W)
        TRANSIENT.add(gw_p.numel() * gw_p.element_size())
        LEDGER.reduce_scatter(g_shard, gw_p)
        TRANSIENT.sub(full.numel() * full.element_size() + gw_p.numel() * gw_p.element_size())
        return gx, g_shard / W, None


class ZEmbed(torch.autograd.Function):
    """table[idx] with the table sharded. Backward needs only idx, not the table, so an embedding pays one
    all-gather (forward) where a linear layer pays two -- which is why stage 3 comes in just under 3P."""

    @staticmethod
    def forward(ctx, idx, shard, shape):
        W = dist.get_world_size()
        full = torch.empty(chunk(math.prod(shape), W) * W, dtype=shard.dtype)
        LEDGER.all_gather(full, shard)
        TRANSIENT.add(full.numel() * full.element_size())
        out = full[:math.prod(shape)].view(shape)[idx]
        ctx.save_for_backward(idx, shard)
        ctx.shape = shape
        TRANSIENT.sub(full.numel() * full.element_size())
        del full
        return out

    @staticmethod
    def backward(ctx, gout):
        idx, shard = ctx.saved_tensors
        shape = ctx.shape
        W = dist.get_world_size()
        g = torch.zeros(shape, dtype=torch.float32).index_add_(0, idx.reshape(-1), gout.reshape(-1, shape[1]).float())
        g_p = padded(g.to(shard.dtype), W)
        TRANSIENT.add(g_p.numel() * g_p.element_size())
        g_shard = torch.empty_like(shard)
        LEDGER.reduce_scatter(g_shard, g_p)
        TRANSIENT.sub(g_p.numel() * g_p.element_size())
        return None, g_shard / W, None


# ============================================================ forward, one code path for every stage
def rms(x):
    return (x.float() * x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()).to(x.dtype)


def forward(get, emb, lin, idx, d, layers, heads):
    """`emb(name, idx)` and `lin(name, x)` hide where the weights live; the model does not know."""
    B, T = idx.shape
    x = emb("emb", idx) + emb("pos", torch.arange(T).expand(B, T))
    for l in range(layers):
        h = rms(x)
        q, k, v = lin(f"l{l}.qkv", h).split(d, dim=-1)
        q, k, v = (t.view(B, T, heads, d // heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + lin(f"l{l}.proj", a)
        x = x + lin(f"l{l}.fc2", F.gelu(lin(f"l{l}.fc1", rms(x))))
    return lin("head", rms(x))


# ============================================================ one rank
def adam(master, m, v, g, t):
    """Adam on fp32 tensors, in place. `g` is the averaged gradient for exactly the elements in `master`."""
    b1, b2 = BETAS
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    master.add_(-LR * mh / (vh.sqrt() + EPS))


def worker(rank, W, stage, steps, port, out):
    global LEDGER, TRANSIENT
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=W)
    LEDGER, TRANSIENT = Ledger(W), Transient()

    ids, V = encode(load_text())
    d, layers, heads, T, B = CFG["d"], CFG["layers"], CFG["heads"], CFG["T"], CFG["B"]
    shapes = param_shapes(V, d, layers)
    init = init_params(shapes)
    N = sum(math.prod(s) for s in shapes.values())
    lo = {k: rank * chunk(math.prod(s), W) for k, s in shapes.items()}
    n_sh = {k: chunk(math.prod(s), W) for k, s in shapes.items()}

    # --- state each rank holds, by stage
    if stage <= 2:   # full bf16 weights on every rank
        P = {k: t.to(torch.bfloat16).requires_grad_(True) for k, t in init.items()}
    else:            # stage 3: this rank's shard of each weight only
        P = {k: padded(t, W)[lo[k]:lo[k] + n_sh[k]].to(torch.bfloat16).requires_grad_(True) for k, t in init.items()}
    if stage == 0:   # full fp32 master + Adam state
        master = {k: t.clone() for k, t in init.items()}
    else:            # shard of fp32 master + Adam state
        master = {k: padded(t, W)[lo[k]:lo[k] + n_sh[k]].clone() for k, t in init.items()}
    m = {k: torch.zeros_like(t) for k, t in master.items()}
    v = {k: torch.zeros_like(t) for k, t in master.items()}
    gshard = {k: torch.zeros(n_sh[k], dtype=torch.bfloat16) for k in shapes} if stage == 2 else None

    if stage == 2:
        # reduce-scatter each gradient the moment backward finishes accumulating it, then drop it,
        # so the full set of gradients never exists on this rank at once
        def hook(name):
            def f(p):
                full = padded(p.grad, W)
                TRANSIENT.add(full.numel() * full.element_size())
                LEDGER.reduce_scatter(gshard[name], full)
                gshard[name].div_(W)
                TRANSIENT.sub(full.numel() * full.element_size())
                p.grad = None
            return f
        for k, p in P.items():
            p.register_post_accumulate_grad_hook(hook(k))

    if stage <= 2:
        emb = lambda name, i: F.embedding(i, P[name])
        lin = lambda name, x: x @ P[name].t()
    else:
        emb = lambda name, i: ZEmbed.apply(i, P[name], shapes[name])
        lin = lambda name, x: ZLinear.apply(x, P[name], shapes[name])

    losses, times = [], []
    for step in range(1, steps + 1):
        idx, tgt = global_batch(ids, step, W * B, T)
        idx, tgt = idx[rank * B:(rank + 1) * B], tgt[rank * B:(rank + 1) * B]
        LEDGER.on = step == 2                                  # count one step's traffic, after warm-up
        c0, t0 = LEDGER.seconds, time.perf_counter()

        logits = forward(P, emb, lin, idx, d, layers, heads)
        loss = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1))
        loss.backward()
        t_bw = time.perf_counter()

        # --- gradients: from local to averaged, in whatever shape this stage keeps them
        if stage == 0:
            for k, p in P.items():
                LEDGER.all_reduce(p.grad)
                p.grad.div_(W)
        elif stage == 1:
            g1 = {}
            for k, p in P.items():
                full = padded(p.grad, W)
                g1[k] = torch.empty(n_sh[k], dtype=torch.bfloat16)
                LEDGER.reduce_scatter(g1[k], full)
                g1[k].div_(W)
        comm_grad = LEDGER.seconds
        t_opt = time.perf_counter()

        # --- the optimizer step, on everything (stage 0) or on this rank's shard (stages 1-3)
        with torch.no_grad():
            for k in shapes:
                if stage == 0:
                    g = P[k].grad.float()
                elif stage == 1:
                    g = g1[k].float()
                elif stage == 2:
                    g = gshard[k].float()
                else:
                    g = P[k].grad.float()
                adam(master[k], m[k], v[k], g, step)
        t_opt_end = time.perf_counter()

        # --- put the weights back where the next forward expects them
        with torch.no_grad():
            if stage == 0:
                for k in shapes:
                    P[k].copy_(master[k].to(torch.bfloat16))
            elif stage in (1, 2):                              # all-gather the updated shards into full weights
                for k, s in shapes.items():
                    full = torch.empty(n_sh[k] * W, dtype=torch.bfloat16)
                    LEDGER.all_gather(full, master[k].to(torch.bfloat16))
                    P[k].copy_(full[:math.prod(s)].view(s))
            else:                                              # stage 3: the shard IS the weight; nothing to send
                for k in shapes:
                    P[k].copy_(master[k].to(torch.bfloat16))
        t_end = time.perf_counter()

        # --- zero the gradients the way each stage keeps them
        if stage in (0, 1):
            for p in P.values():
                p.grad.zero_()                                 # a full-size buffer that persists across steps
        elif stage == 3:
            for p in P.values():
                p.grad = None                                  # frees the shard-sized gradient
        if step > 1:
            comm = LEDGER.seconds - c0
            times.append(dict(total=t_end - t0, comm=comm, optimizer=t_opt_end - t_opt,
                              compute=(t_end - t0) - comm - (t_opt_end - t_opt)))
        # the global loss, for the correctness check only -- deliberately not on the ledger
        with torch.no_grad():
            gl = loss.detach().clone()
            dist.all_reduce(gl)
            losses.append(gl.item() / W)

    # --- memory: walk the tensors this rank actually holds, after a step, by category
    def nbytes(ts):
        return sum(t.numel() * t.element_size() for t in ts)
    grads_held = ([p.grad for p in P.values() if p.grad is not None] if stage in (0, 1)
                  else list(gshard.values()) if stage == 2
                  else [p.grad for p in P.values() if p.grad is not None])
    if stage == 3:   # stage 3 frees its shard gradients after the step; count what the step needs
        grads_held = [torch.empty(n_sh[k], dtype=torch.bfloat16) for k in shapes]
    mem = dict(weights_bf16=nbytes(P.values()), grads_bf16=nbytes(grads_held),
               master_fp32=nbytes(master.values()), adam_m=nbytes(m.values()), adam_v=nbytes(v.values()))
    per_rank = dict(rank=rank, mem=mem, transient_peak=TRANSIENT.peak, ledger_bytes=LEDGER.bytes,
                    ledger_calls=LEDGER.calls, times=times, losses=losses)
    everyone = [None] * W
    dist.all_gather_object(everyone, per_rank)
    if rank == 0:
        Path(out).write_text(json.dumps(dict(W=W, stage=stage, N=N, V=V, ranks=everyone), default=float))
    dist.destroy_process_group()


# ============================================================ the single-process reference
def reference(steps, W):
    """One process, the whole global batch (W x B sequences), fp32 master + Adam, bf16 weights.
    Data parallelism claims to be exactly this; the loss curves are compared below."""
    torch.set_num_threads(4)
    ids, V = encode(load_text())
    d, layers, heads, T, B = CFG["d"], CFG["layers"], CFG["heads"], CFG["T"], CFG["B"]
    shapes = param_shapes(V, d, layers)
    init = init_params(shapes)
    P = {k: t.to(torch.bfloat16).requires_grad_(True) for k, t in init.items()}
    master = {k: t.clone() for k, t in init.items()}
    m = {k: torch.zeros_like(t) for k, t in init.items()}
    v = {k: torch.zeros_like(t) for k, t in init.items()}
    emb = lambda name, i: F.embedding(i, P[name])
    lin = lambda name, x: x @ P[name].t()
    losses = []
    for step in range(1, steps + 1):
        idx, tgt = global_batch(ids, step, W * B, T)
        # the mean over W slices of B sequences = the mean over all W*B sequences (equal token counts)
        loss = sum(F.cross_entropy(forward(P, emb, lin, idx[r * B:(r + 1) * B], d, layers, heads).float().reshape(-1, V),
                                   tgt[r * B:(r + 1) * B].reshape(-1)) for r in range(W)) / W
        loss.backward()
        with torch.no_grad():
            for k in shapes:
                adam(master[k], m[k], v[k], P[k].grad.float(), step)
                P[k].copy_(master[k].to(torch.bfloat16))
                P[k].grad.zero_()
        losses.append(loss.item())
    return losses


# ============================================================ driver
def run(W, stage, steps, port):
    out = HERE / f".run_W{W}_s{stage}.json"
    mp.spawn(worker, args=(W, stage, steps, port, str(out)), nprocs=W, join=True)
    R = json.loads(out.read_text())
    out.unlink()
    return R


def summarise(R):
    W, N = R["W"], R["N"]
    P_bytes = 2 * N                                              # the model in bf16
    r0 = R["ranks"][0]
    mem = {k: max(r["mem"][k] for r in R["ranks"]) for k in r0["mem"]}   # the fullest rank
    persistent = sum(mem.values())
    tr = max(r["transient_peak"] for r in R["ranks"])
    traffic = {k: v / P_bytes for k, v in r0["ledger_bytes"].items()}
    ts = [t for r in R["ranks"] for t in r["times"]]
    avg = {k: sum(t[k] for t in ts) / len(ts) for k in ("total", "compute", "comm", "optimizer")}
    return dict(W=W, stage=R["stage"], N=N, mem=mem, persistent=persistent, bytes_per_param=persistent / N,
                transient_peak=tr, traffic_P=traffic, traffic_total_P=sum(traffic.values()),
                calls=r0["ledger_calls"], time=avg, losses=r0["losses"])


def theory(stage, W):
    """bytes per parameter per rank, from the brief's Section 6 table, at world size W."""
    return {0: 16.0, 1: 4 + 12 / W, 2: 2 + 14 / W, 3: 16 / W}[stage]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.replot:
        plot_all(json.loads((HERE / "results.json").read_text()))
        return
    load_text()
    worlds = [8] if args.quick else [8, 16, 32]
    steps = 6 if args.quick else 12
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(s)
    say(f"ERA V5 Session 12 -- ZeRO by hand on {max(worlds)} virtual GPUs (gloo, one CPU process each) | seed {SEED} | torch {torch.__version__}")
    say(f"model: tiny char-level GPT, d={CFG['d']}, {CFG['layers']} layers, {CFG['heads']} heads, T={CFG['T']}; "
        f"{CFG['B']} sequences per rank per step; {steps} steps; Adam, bf16 weights, fp32 master")
    res, port = [], 29700
    for W in worlds:
        for stage in (0, 1, 2, 3):
            t0 = time.time()
            s = summarise(run(W, stage, steps, port))
            port += 1
            s["wall_s"] = time.time() - t0
            res.append(s)
            say(f"  W={W:>2} stage {stage}: {s['bytes_per_param']:5.2f} B/param per rank (theory {theory(stage, W):5.2f}), "
                f"traffic {s['traffic_total_P']:.3f} P/step, step {1000 * s['time']['total']:6.1f} ms "
                f"(comm {1000 * s['time']['comm']:5.1f}, optimizer {1000 * s['time']['optimizer']:5.2f}), "
                f"loss@{steps} {s['losses'][-1]:.4f}")
    Wmax = max(worlds)
    ref = reference(steps, Wmax)
    top = [s for s in res if s["W"] == Wmax]
    diffs = {s["stage"]: max(abs(a - b) for a, b in zip(s["losses"], ref)) for s in top}
    say("")
    say(f"  reference: one process, the whole {Wmax * CFG['B']}-sequence batch: loss@{steps} {ref[-1]:.4f}")
    say("  largest loss difference from the reference over the run: "
        + ", ".join(f"stage {k} {v:.1e}" for k, v in diffs.items()))
    N = res[0]["N"]
    R = dict(config=CFG, seed=SEED, steps=steps, worlds=worlds, N=N, P_bytes=2 * N, runs=res,
             reference_losses=ref, loss_diff_vs_reference=diffs, torch=torch.__version__,
             cpu_threads=os.cpu_count())
    (HERE / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (HERE / "run.log").write_text("\n".join(log) + "\n")
    plot_all(R)


# ============================================================ figures
def plot_all(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(parents=True, exist_ok=True)
    Wmax = max(R["worlds"])
    top = sorted((s for s in R["runs"] if s["W"] == Wmax), key=lambda s: s["stage"])
    N = R["N"]

    def style(ax, title, ylabel, xlabel=None):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, color=INK2, fontsize=9)

    # 1. memory per rank, by what it is, at the largest world size
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    cats = [("weights_bf16", "weights (bf16)", BLUE), ("grads_bf16", "gradients (bf16)", ORANGE),
            ("opt", "optimizer: fp32 copy + Adam m, v", AQUA)]
    xs = list(range(len(top)))
    bottom = [0.0] * len(top)
    for key, label, col in cats:
        vals = [(s["mem"]["master_fp32"] + s["mem"]["adam_m"] + s["mem"]["adam_v"]) / N if key == "opt"
                else s["mem"][key] / N for s in top]
        ax.bar(xs, vals, bottom=bottom, color=col, width=0.6, label=label, edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, s in zip(xs, top):
        ax.text(x, s["bytes_per_param"] + 0.3, f"{s['bytes_per_param']:.2f} B", ha="center", color=INK, fontsize=10)
    ax.set_xticks(xs, [f"stage {s['stage']}" + (" (DP)" if s["stage"] == 0 else "") for s in top])
    style(ax, f"Memory per virtual GPU, bytes per parameter, {Wmax} GPUs (measured from the tensors held)", "bytes per parameter")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "memory_by_stage.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 2. where one step's time goes, at the largest world size
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    bottom = [0.0] * len(top)
    for key, label, col in (("compute", "forward + backward", BLUE), ("comm", "inside collectives", ORANGE),
                            ("optimizer", "optimizer step", AQUA)):
        vals = [1000 * s["time"][key] for s in top]
        ax.bar(xs, vals, bottom=bottom, color=col, width=0.6, label=label, edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, s in zip(xs, top):
        ax.text(x, 1000 * s["time"]["total"] * 1.02, f"{s['traffic_total_P']:.2f} P sent", ha="center", color=INK, fontsize=9)
    ax.set_xticks(xs, [f"stage {s['stage']}" for s in top])
    style(ax, f"One training step, per virtual GPU, {Wmax} GPUs on {R['cpu_threads']} CPU threads", "milliseconds")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "time_by_stage.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 3. measured bytes per parameter against world size, one line per stage, theory dashed
    if len(R["worlds"]) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
        cols = [BLUE, ORANGE, AQUA, YELLOW]
        for st in range(4):
            rs = sorted((s for s in R["runs"] if s["stage"] == st), key=lambda s: s["W"])
            ws = [s["W"] for s in rs]
            ax.plot(ws, [s["bytes_per_param"] for s in rs], color=cols[st], lw=2, marker="o", ms=7)
            ax.plot(ws, [theory(st, w) for w in ws], color=cols[st], lw=1, ls="--")
            ax.text(ws[-1] * 1.05, rs[-1]["bytes_per_param"], f"stage {st}", color=INK, va="center", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(R["worlds"], [str(w) for w in R["worlds"]])
        ax.set_xlim(R["worlds"][0] * 0.8, R["worlds"][-1] * 1.6)
        style(ax, "Bytes per parameter per GPU (dots measured; dashes = formula, hidden under them)",
              "bytes per parameter", "number of virtual GPUs")
        fig.tight_layout()
        fig.savefig(FIGDIR / "memory_vs_world.png", dpi=130, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    main()
