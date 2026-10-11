#!/usr/bin/env python3
"""Build stability.ipynb — a short live version of the harness.

It imports the same modules the real run used, so the notebook cannot disagree with
run.log. Stages 1 and 2 run in full because they are numpy and cheap; stages 3 to 5 run
at a reduced size.

    python make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace stability.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-15-stability"
_ID = [0]


def _id():
    _ID[0] += 1
    return f"cell-{_ID[0]:02d}"


def md(*l):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": "\n".join(l)}


def code(*l):
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {},
            "outputs": [], "source": "\n".join(l)}


CELLS = [
    md("# ERA V5 · Session 15 — a training run that watches itself",
       "",
       "Session 15's assignment is a team-selection form, so this notebook answers the",
       "session's own open questions instead. Four published spike rules are scored on one",
       "trace; M4 downsampling is checked pixel for pixel; a silent checkpoint misload is",
       "reproduced and its signal measured down to the noise floor; and the freezing",
       "constraints that break without an error are made to break.",
       "",
       "Full-length numbers are in `README.md` and `results.json`; the log is `run.log`."),

    code("import os, sys, json",
         "if not os.path.exists('detect.py'):              # on Colab: fetch the code",
         f"    !git clone -q {REPO} 2>/dev/null || true",
         f"    os.chdir('{DIR}')",
         "sys.path.insert(0, os.getcwd())",
         "sys.path.insert(0, os.path.join(os.getcwd(), '..', 'session-14-moe'))",
         "import numpy as np, detect, rollup",
         "print('numpy', np.__version__)"),

    md("## 1. The trace, and the spikes we planted",
       "",
       "A rule can only be scored against a trace whose spikes are known. This one has a",
       "warmup, a decaying level, lognormal noise whose spread falls with the level, and 40",
       "planted spikes of known height and position."),
    code("STEPS = 300_000",
         "v, pos, heights = detect.trace(STEPS)",
         "print(f'{STEPS:,} steps · median {np.median(v):.3f} · max {v.max():.2f} "
         "· {len(pos)} spikes planted')",
         "print('first five spike steps', pos[:5], 'heights', np.round(heights[:5], 1))"),

    md("## 2. Score all four rules on it",
       "",
       "A spike counts as found if an alarm lands within 4 steps of it. Watch the false-alarm",
       "column, not the found column — every rule but one finds the spikes."),
    code("rules = [('fixed 5.0', lambda v: detect.rule_fixed(v, 5.0)),",
         "         ('OLMo 2 spike score', detect.rule_spike_score),",
         "         ('ZClip EMA z (2.5)', detect.rule_ema_z),",
         "         ('Megatron ratio to max', detect.rule_ratio_max)]",
         "print(f\"{'rule':24s} {'found':>6s} {'missed':>7s} {'false':>8s} {'false/10k':>10s}\")",
         "for name, fn in rules:",
         "    s = detect.score(fn(v), pos)",
         "    print(f'{name:24s} {s[\"found\"]:>6d} {s[\"missed\"]:>7d} "
         "{s[\"false_alarms\"]:>8,} {detect.per_10k(s, STEPS):>10.2f}')"),

    md("### Where should the EMA threshold sit?",
       "",
       "ZClip's 2.5 is a *clipping* threshold — the method rescales the gradient, it does not",
       "page anyone. Used as an alert it fires constantly. The detection rate does not fall",
       "until well past z = 5, because a real spike is many sigma out."),
    code("print(f\"{'z':>5s} {'found':>6s} {'false':>8s} {'false/10k':>10s}\")",
         "for z in (2.5, 3.0, 4.0, 5.0, 6.0, 8.0):",
         "    s = detect.score(detect.rule_ema_z(v, z=z), pos)",
         "    print(f'{z:>5.1f} {s[\"found\"]:>6d} {s[\"false_alarms\"]:>8,} "
         "{detect.per_10k(s, STEPS):>10.2f}')"),

    md("## 3. Does M4 downsampling really lose nothing?",
       "",
       "300,000 steps on a 2,000-pixel chart is 150 steps a column. Each method is rasterised",
       "and compared **pixel for pixel** against the full-data rendering — a stronger test",
       "than looking at the curve. The bucket boundaries must use the same index-to-column",
       "mapping the renderer uses, or M4's proof does not apply."),
    code("y, _, _ = detect.loss_trace(STEPS)",
         "W, H = 2000, 600",
         "lo, hi = float(y.min()), float(y.max())",
         "gf = rollup.raster(np.stack([np.arange(len(y)), y], 1), len(y), W, H, lo, hi)",
         "cases = [('mean', rollup.ds_mean(y, W)),",
         "         ('min/max', rollup.ds_minmax(y, W)),",
         "         ('M4', rollup.ds_m4(y, W)),",
         "         ('LTTB', rollup.ds_lttb(y, W)),",
         "         ('min/max per column', rollup.ds_minmax(y, W, per_column=True)),",
         "         ('M4 per column', rollup.ds_m4(y, W, per_column=True))]",
         "print(f\"{'method':22s} {'points':>7s} {'wrong px':>9s} {'peak err':>10s}\")",
         "for name, p in cases:",
         "    g = rollup.raster(p, len(y), W, H, lo, hi)",
         "    print(f'{name:22s} {len(p):>7,} {int((g ^ gf).sum()):>9,} "
         "{abs(p[:,1].max() - y.max()):>10.4f}')"),

    md("## 4. The silent checkpoint misload",
       "",
       "A converter writes tensors under the source model's key names; the target wraps its",
       "stack differently; `strict=False` skips them and reports success. Every cheap check",
       "passes. The only signal is the loss at step one — and the question is how small that",
       "gets when only *part* of the model is misnamed.",
       "",
       "Needs `session-14-moe/dense.pt`, which Session 14's `train.py` writes."),
    code("import misload",
         "out = misload.main()",
         "print()",
         "print('gap at 2% misnamed', out['sweep'][-1]['gap'],",
         "      'vs 2 sigma of batch noise', round(2 * out['batch_noise_sigma'], 4))"),

    md("## 5. Freezing, and the four things that break quietly",
       "",
       "The stale gradient is the whole mechanism: PyTorch skips a parameter whose `grad` is",
       "`None`, so freezing alone is safe — but a zero gradient left over from before the",
       "freeze lets AdamW's decoupled weight decay keep shrinking it, with no gradient",
       "involved. The loss cannot see it. A hash can."),
    code("import freeze",
         "fz = freeze.main()"),

    md("## 6. The full-length numbers",
       "",
       "Read from `results.json`, which came from `python run.py`."),
    code("R = json.loads(open('results.json').read())",
         "print('device', R.get('gpu'), '· stages', R.get('stages'))",
         "for r in R['detectors']['grad_norm']:",
         "    print(f\"  {r['rule']:38s} found {r['found']:2d}  false/10k {r['per_10k']:8.2f}\")",
         "rt = R.get('routing')",
         "if rt and rt.get('arms'):",
         "    ceil = rt['arms'][0]['per_layer'][0]['intervention_ceiling']",
         "    unif = rt['arms'][0]['per_layer'][0]['uniform_topk_mass']",
         "    print(f'  per-slot intervention ceiling 1-k/N = {ceil:.4f} "
         "· uniform top-k mass k/N = {unif:.4f}')",
         "    for a in rt['arms']:",
         "        iv = [p['intervention_rate'] for p in a['per_layer']]",
         "        sel = [p['selected_topk_mass'] for p in a['per_layer']]",
         "        print(f\"  balance={a['balance']:5s} MaxVio {a['mean_maxvio']:.4f} \"",
         "              f\"dead {a['dead_total']:3d} interv/slot {min(iv):.3f}-{max(iv):.3f} \"",
         "              f\"min selected mass {min(sel):.3f}\")"),

    md("## What to take away",
       "",
       "- **A rule that finds every spike is not therefore a good alarm.** The number that",
       "  decides whether an alert channel gets read is the false-alarm rate, and only one of",
       "  the four published rules is usable as written.",
       "- **M4 is exact, and only in its published form.** Four points per pixel column, with",
       "  buckets aligned to the renderer's own mapping. Min and max alone are not enough.",
       "- **A silent misload is invisible below about 5% of tensors**, and a partial one is",
       "  worse than a total one, because the correctly loaded parts drive confident wrong",
       "  logits. The fix is structural: build the checkpoint from the target keyspace and",
       "  require a strict load.",
       "- **Freezing changes the grad norm**, so the clip limit and the alarm band both have",
       "  to move with it. And which side you freeze decides the memory, because the experts",
       "  hold most of the parameters.",
       "- Every one of these failures passes a finite-loss check. That is the session's whole",
       "  point: for each assumption the run depends on, log the number that proves it."),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "stability.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out.name} -- {len(CELLS)} cells")
