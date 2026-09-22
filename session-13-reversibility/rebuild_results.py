"""Rebuild results.json from run.log, then re-run only the cheap measurement that crashed.

The first full run finished every training run and then died in the memory-against-depth sweep,
which ran out of memory at 24 layers and took the script down before anything had been written.
The training numbers survived in run.log, so they are parsed back out here rather than spending
another three GPU-hours reproducing them. train.py now saves after every stage and tolerates an
out-of-memory error in the sweep, so this script is a one-off repair.

Every field that cannot be recovered from the log is omitted rather than invented. `seconds` is
derived from tokens ÷ tokens-per-second, which is exact, and the loss curves are gone.

    python rebuild_results.py
"""
import json
import re
from pathlib import Path

import torch

import train as T

HERE = Path(__file__).resolve().parent
log = (HERE / "run.log").read_text()

PARAMS, BLOCK, TOKENS = 21_244_416, 512, 50_000_000


def done_lines():
    """[tag] done: val X · Y tok/s · Z MiB peak"""
    pat = r"\[([^\]]+)\] done: val ([\d.]+) · ([\d,]+) tok/s · ([\d,]+) MiB peak"
    return [(t, float(v), float(s.replace(",", "")), float(m.replace(",", "")))
            for t, v, s, m in re.findall(pat, log)]


def steps_of(tag):
    m = re.findall(rf"\[{re.escape(tag)}\] step \d+/(\d+)", log)
    return int(m[-1]) if m else None


def row(tag, stack, val, tps, mem, tokens):
    steps = steps_of(tag)
    batch = round(tokens / (steps * BLOCK)) if steps else None
    return dict(tag=tag, stack=stack, batch=batch, steps=steps, tokens=tokens, params=PARAMS,
                h=0.25, a=1.0, layers=12, val_loss=round(val, 4),
                final_train_loss=None, seconds=round(tokens / tps, 1), tokens_per_s=tps,
                peak_mem_mib=mem, curve=None,
                recovered_from="run.log (the first run died in the depth sweep before writing JSON)")


R = dict(device=torch.cuda.get_device_name(0) if T.DEV == "cuda" else "cpu",
         torch=torch.__version__, seed=T.SEED, block=BLOCK, tokens=TOKENS, quick=False)

R["grad_check"] = [dict(stack=s, worst_rel_grad_error=float(e)) for s, e in
                   re.findall(r"^  (\w+)\s+worst relative gradient error ([\d.e+-]+)$", log, re.M)]
R["reconstruction"] = [dict(stack=s, a=float(a), rel_error=float(e)) for s, a, e in
                       re.findall(r"^  (\w+)\s+a=([\d.]+)\s+relative error ([\d.e+-]+)$", log, re.M)]
R["max_batch"] = {s: int(b) for s, b in
                  re.findall(r"^  (baseline|midpoint|leapfrog|hamiltonian)\s+(\d+)$", log, re.M)}

VARIANT_TOKENS = 4_000_000
R["variants"], R["runs"] = [], []
for tag, val, tps, mem in done_lines():
    if tag.startswith("variant:"):
        R["variants"].append(row(tag, tag.split(":")[1], val, tps, mem, VARIANT_TOKENS))
    else:
        stack = "baseline" if tag == "1-baseline" else "leapfrog"
        R["runs"].append(row(tag, stack, val, tps, mem, TOKENS))
R["best_variant"] = min(R["variants"], key=lambda r: r["val_loss"])["stack"]

print(f"recovered {len(R['variants'])} variant runs and {len(R['runs'])} training runs from run.log")
for r in R["runs"]:
    print(f"  {r['tag']:26s} batch {r['batch']:>4} · val {r['val_loss']:.4f} · "
          f"{r['tokens_per_s']:,.0f} tok/s · {r['peak_mem_mib']:,.0f} MiB")

print("\n== memory against depth (re-run; the sweep that crashed)", flush=True)
R["memory_vs_depth"] = T.memory_vs_depth(max(1, R["max_batch"]["baseline"] // 2), [4, 8, 12, 24, 48])

(HERE / "results.json").write_text(json.dumps(R, indent=1))
print("\nwrote results.json")
