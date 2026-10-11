#!/usr/bin/env python3
"""Session 15 — the stability, debugging and monitoring harness, run end to end.

Five stages, each one answering a question the session leaves open. Every stage writes
results.json before the next one starts, so a crash costs one stage.

    1. detectors      four published spike rules, scored on one trace with planted spikes
    2. downsampling   mean / min-max / M4 / LTTB against the full rendering, in pixels
    3. misload        the silent checkpoint misload, and how small its only signal gets
    4. freezing       the four constraints that break without an error
    5. routing        the five per-layer metrics, and the fault only they can see

    python run.py              # the full run
    python run.py --quick      # minutes, for a smoke test
    python run.py --only 1,2   # selected stages
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "session-14-moe"))

R = {}
LOG = HERE / "run.log"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with LOG.open("a") as f:
        f.write(s + "\n")


def save():
    (HERE / "results.json").write_text(json.dumps(R, indent=2))


# ------------------------------------------------------------------ 1. detectors

def stage_detectors(quick):
    import detect

    steps = 30_000 if quick else detect.STEPS
    log(f"\n== 1. spike detectors · {steps:,} steps, 40 planted spikes")
    R["detectors"] = {}
    for curve in ("grad_norm", "loss"):
        v, pos, heights = (detect.trace(steps) if curve == "grad_norm"
                           else detect.loss_trace(steps))
        rules = [
            ("fixed 5.0", lambda v: detect.rule_fixed(v, 5.0)),
            ("OLMo 2 spike score (7 sigma / 1,000)", detect.rule_spike_score),
            ("ZClip EMA z (alpha 0.97, z 2.5)", detect.rule_ema_z),
            ("Megatron ratio to max (10x)", detect.rule_ratio_max),
        ]
        rows = []
        log(f"\n   on the {curve}: median {float(__import__('numpy').median(v)):.3f}, "
            f"max {v.max():.2f}")
        log(f"   {'rule':38s} {'found':>6s} {'missed':>7s} {'false':>7s} {'false/10k':>10s}")
        for name, fn in rules:
            t0 = time.time()
            s = detect.score(fn(v), pos)
            s.update(rule=name, curve=curve, per_10k=detect.per_10k(s, steps),
                     seconds=round(time.time() - t0, 2))
            rows.append(s)
            log(f"   {name:38s} {s['found']:>6d} {s['missed']:>7d} "
                f"{s['false_alarms']:>7d} {s['per_10k']:>10.2f}")
        R["detectors"][curve] = rows

        # Where does the EMA rule become pageable? Sweep z.
        if curve == "grad_norm":
            sweep = []
            log(f"\n   ZClip EMA z-score, sweeping the threshold")
            log(f"   {'z':>5s} {'found':>6s} {'missed':>7s} {'false':>7s} {'false/10k':>10s}")
            for z in (2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
                s = detect.score(detect.rule_ema_z(v, z=z), pos)
                s.update(z=z, per_10k=detect.per_10k(s, steps))
                sweep.append(s)
                log(f"   {z:>5.1f} {s['found']:>6d} {s['missed']:>7d} "
                    f"{s['false_alarms']:>7d} {s['per_10k']:>10.2f}")
            R["detectors"]["ema_z_sweep"] = sweep
    R["detectors"]["steps"] = steps
    save()


# ------------------------------------------------------------------ 2. downsampling

def stage_downsample(quick):
    import numpy as np

    import detect
    import rollup

    steps = 30_000 if quick else detect.STEPS
    W, H = 2_000, 600
    y, pos, _ = detect.loss_trace(steps)
    lo, hi = float(y.min()), float(y.max())
    full = np.stack([np.arange(len(y)), y], 1)
    gf = rollup.raster(full, len(y), W, H, lo, hi)

    log(f"\n== 2. downsampling · {steps:,} steps onto {W} pixels "
        f"({steps / W:.0f} steps a column), rasterised {W}x{H}")
    rows = []
    cases = [
        ("mean", rollup.ds_mean(y, W), "one value a column"),
        ("min/max", rollup.ds_minmax(y, W), "same point budget"),
        ("M4", rollup.ds_m4(y, W), "same point budget"),
        ("LTTB", rollup.ds_lttb(y, W), "same point budget"),
        ("min/max per column", rollup.ds_minmax(y, W, per_column=True), "2 a column"),
        ("M4 per column", rollup.ds_m4(y, W, per_column=True), "<=4 a column, as published"),
    ]
    log(f"   {'method':22s} {'points':>7s} {'wrong px':>9s} {'of':>9s} "
        f"{'peak error':>11s} {'note'}")
    for name, pts, note in cases:
        g = rollup.raster(pts, len(y), W, H, lo, hi)
        wrong = int((g ^ gf).sum())
        peak = float(abs(pts[:, 1].max() - y.max()))
        rows.append(dict(method=name, points=int(len(pts)), wrong_pixels=wrong,
                         of_pixels=int(gf.sum()), peak_error=round(peak, 4), note=note))
        log(f"   {name:22s} {len(pts):>7,} {wrong:>9,} {int(gf.sum()):>9,} "
            f"{peak:>11.4f}  {note}")
    R["downsampling"] = dict(steps=steps, width=W, height=H, rows=rows,
                             spikes_planted=int(len(pos)))
    save()


# ------------------------------------------------------------------ 3. misload

def stage_misload(quick):
    import misload

    log("\n== 3. the silent checkpoint misload")
    out = misload.main()
    R["misload"] = out
    save()


# ------------------------------------------------------------------ 4. freezing

def stage_freeze(quick):
    import freeze

    log("\n== 4. freezing: the constraints that break without an error")
    R["freezing"] = freeze.main()
    save()


# ------------------------------------------------------------------ 5. routing health

def stage_routing(quick, tokens_override=0):
    import torch

    import model as M
    import routing
    import train as T

    dense_pt = HERE.parent / "session-14-moe/dense.pt"
    if not dense_pt.exists():
        log(f"   SKIPPED: {dense_pt} is missing -- run Session 14's train.py first")
        R["routing"] = dict(skipped=str(dense_pt))
        save()
        return

    tokens = tokens_override or (400_000 if quick else 4_000_000)
    # The GPU is the only thing that makes the large batch worth it; on CPU the dispatch
    # loop dominates and a batch of 4 is as fast per token as a batch of 8.
    cpu = T.DEV == "cpu"
    batch, accum = (4, 1) if (quick or cpu) else (8, 2)
    iters = 20 if (quick or cpu) else 60
    dense = M.GPT(M.Config())
    dense.load_state_dict(torch.load(dense_pt, map_location="cpu"))
    log(f"\n== 5. routing health · {tokens:,} tokens a run, device {T.DEV}")

    R["routing"] = dict(tokens=tokens, batch=batch, accum=accum, arms=[])
    for bal in ("bias", "none"):
        cfg = M.Config(moe=True, balance=bal)
        net = M.upcycle(dense, "partition", cfg, scale_down=True)
        row = T.run(net, f"route:{bal}", tokens, batch, accum)
        va = T.Batches("val", batch)
        per_layer = routing.measure(net, va, iters, cfg.top_k)
        base, abl = routing.ablate_layers(net, va, max(iters // 3, 8), cfg.top_k)

        log(f"\n   balance={bal}  val {row['val_loss']:.4f}  "
            f"mean MaxVio {sum(p['maxvio'] for p in per_layer) / len(per_layer):.4f}")
        log(f"   {'layer':>5s} {'MaxVio':>8s} {'dead':>5s} {'entropy':>8s} "
            f"{'interv':>8s} {'intr mass':>10s} {'sel mass':>9s} "
            f"{'random':>8s} {'bypass':>8s}")
        by_layer = {a["layer"]: a for a in abl}
        for i, p in enumerate(per_layer):
            a = by_layer.get(i, {})
            log(f"   {i:>5d} {p['maxvio']:>8.4f} {p['dead']:>5d} "
                f"{p['router_entropy']:>8.4f} {p['intervention_rate']:>8.4f} "
                f"{p['intrinsic_topk_mass']:>10.4f} {p['selected_topk_mass']:>9.4f} "
                f"{a.get('random_delta', float('nan')):>+8.4f} "
                f"{a.get('bypass_delta', float('nan')):>+8.4f}")
        log(f"   ceiling for the intervention rate, 1 - k/N = "
            f"{per_layer[0]['intervention_ceiling']:.4f}; "
            f"uniform top-k mass = k/N = {per_layer[0]['uniform_topk_mass']:.4f}")

        R["routing"]["arms"].append(dict(
            balance=bal, run=row, base_loss=round(base, 4),
            per_layer=per_layer, ablation=abl,
            mean_maxvio=round(sum(p["maxvio"] for p in per_layer) / len(per_layer), 4),
            dead_total=sum(p["dead"] for p in per_layer)))
        del net
        if T.DEV == "cuda":
            torch.cuda.empty_cache()
        save()


STAGES = {1: stage_detectors, 2: stage_downsample, 3: stage_misload,
          4: stage_freeze, 5: stage_routing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--tokens", type=int, default=0,
                    help="override stage 5's token budget per arm")
    a = ap.parse_args()
    want = ([int(x) for x in a.only.split(",")] if a.only else sorted(STAGES))

    # A selective re-run keeps the results of the stages it is not running, and appends
    # to the log, so the five stages can be run in any order or on different hardware.
    out = HERE / "results.json"
    if a.only and out.exists():
        R.update(json.loads(out.read_text()))
    else:
        LOG.write_text("")
    import torch

    import train as T
    R.update(quick=a.quick and not a.only, torch=torch.__version__, device=T.DEV,
             gpu=torch.cuda.get_device_name(0) if T.DEV == "cuda" else "cpu",
             stages=want, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"Session 15 stability harness · torch {torch.__version__} · {R['gpu']} "
        f"· {'quick' if a.quick else 'full'} · stages {want}")
    save()
    t0 = time.time()
    for s in want:
        if s == 5:
            STAGES[s](a.quick, a.tokens)
        else:
            STAGES[s](a.quick)
    R["seconds"] = round(time.time() - t0, 1)
    log(f"\ndone in {R['seconds']:,.0f}s")
    save()


if __name__ == "__main__":
    main()
