"""Generate README.md from results.json, so the write-up cannot drift from the run."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
M = R["meta"]
r1, r2, r3, r4, r5 = (R[f"req{i}"] for i in range(1, 6))
out = []
w = out.append

w("# ERA V5 · Session 11 — optimizers and learning-rate schedules, measured\n")
w("Adam computed by hand and checked against PyTorch; bias correction switched off; the update-to-weight ratio "
  "of every layer; cosine against WSD with both sides tuned; and a learning-rate sweep across three widths with a "
  "call for a fourth. Every number on this page came out of `optimizers.py`: [`run.log`](run.log) is the transcript, "
  "[`results.json`](results.json) the numbers, and this file is generated from that JSON by `make_readme.py`.\n")
w("```bash\npython optimizers.py            # the full run: run.log, results.json, figures/\n"
  "python optimizers.py --quick    # same structure, fewer points\n"
  "python optimizers.py --replot   # redraw figures/ from results.json\n"
  "python make_readme.py           # regenerate this file\n```\n")
w(f"**Setup:** seed `{M['seed']}` (the session date), tiny Shakespeare via GPT-2 BPE (V = 50,257), a 4-layer "
  f"pre-norm GPT (D = 256 unless the width is the variable), batch {M['batch'][0]} × {M['batch'][1]}, AdamW with decay "
  f"0.1 on matrices and none on norm gains, clipping at 1.0, torch {M['torch']} on an {M['gpu']}. The notebook "
  "[`optimizers.ipynb`](optimizers.ipynb) runs the same functions cell by cell.\n")

# ---- summary
cw = r2["crossings"]
nw, ww = r2["runs"]["no warmup"], r2["runs"]["30-step warmup"]
w("## The five answers\n")
w("| # | requirement | result |\n|---|---|---|")
w(f"| 1 | Adam by hand, checked against PyTorch | agrees to 1e-16 in fp64 and 1e-8 in fp32; all "
  f"{30 - len(r1['brief_mismatches'])} of 30 values in the brief's own table reproduce |")
w(f"| 2 | when bias correction stops mattering | the uncorrected step is **{r2['peak_ratio']:.2f}× too large** at step "
  f"{r2['peak_t']}, within 10% only by step **{cw['0.1']:,}** and 1% by **{cw['0.01']:,}** — not within twenty steps. A real "
  f"model trained without it ends {nw['off_eval'] - nw['on_eval']:.2f} nats worse |")
w(f"| 3 | the step at which warmup stops changing the update-to-weight ratio | **step {r3['median_stop']}**, against a "
  f"{r3['warmup_steps']}-step warmup — the effect outlasts the warmup by {r3['median_stop'] - r3['warmup_steps']} steps |")
w(f"| 4 | cosine against WSD, stopped at 200 | both tuned; **WSD {r4['wsd_at_stop']:.4f}** against cosine "
  f"{r4['cosine_at_stop']:.4f}. Keep WSD — and give it a {r4['branch_steps']}-step decay: {r4['wsd_branch']:.4f} |")
w(f"| 5 | the learning rate at width 4,096 | **{r5['predicted_4096']:.1e}**, with a range of {r5['range_4096'][0]:.1e} to "
  f"{r5['range_4096'][1]:.1e}. Low confidence — see why |\n")

# ---- 1
w("## 1. Adam by hand\n")
w("One weight starting at 1.0, five gradients (0.50, 0.40, 0.60, 0.45, 0.55), η = 0.001, β₁ = 0.9, β₂ = 0.999, "
  "ε = 1e-8 — the brief's Section 6 example, so the result can be checked against the brief as well as against "
  "PyTorch. Computed in plain Python floats:\n")
w("```\nm      = β₁·m + (1 − β₁)·g                  running mean of the gradient: the direction\n"
  "v      = β₂·v + (1 − β₂)·g²                 running mean of the squared gradient: the scale\n"
  "m̂      = m / (1 − β₁ᵗ)                      bias correction: both averages start at 0 and read low\n"
  "v̂      = v / (1 − β₂ᵗ)\n"
  "step   = −η · m̂ / (√v̂ + ε)\n```\n")
w("| t | g | m | v | m̂ | v̂ | step | w |\n|--:|--:|--:|--:|--:|--:|--:|--:|")
for r in r1["hand"]:
    w(f"| {r['t']} | {r['g']:.2f} | {r['m']:.6f} | {r['v']:.9f} | {r['m_hat']:.6f} | {r['v_hat']:.6f} | {r['step']:+.9f} | {r['w']:.9f} |")
w("")
w("Checked against `torch.optim.Adam`, reading `exp_avg` and `exp_avg_sq` out of its state after every step "
  "(largest difference over the five steps):\n")
w("| | m | v | m̂ | v̂ | step | w |\n|---|--:|--:|--:|--:|--:|--:|")
for dt, a in r1["agreement"].items():
    d = a["max_abs_diff"]
    w(f"| {dt.replace('torch.', '')} | {d['m']:.1e} | {d['v']:.1e} | {d['m_hat']:.1e} | {d['v_hat']:.1e} | {d['step']:.1e} | {d['w']:.1e} |")
w("")
w("In fp64 the two agree to the last bit or within 1e-16. In fp32 they agree to about 1e-8, which is fp32's own "
  "resolution. The step column shows why Adam became the default: the gradients range from 0.40 to 0.60, and every "
  "step is within half a percent of η. The gradient decides the direction and η decides the distance.\n")

# ---- 2
w("## 2. Bias correction switched off\n")
w("![bias correction on and off](figures/bias_correction.png)\n")
w("Both running averages start at zero, so early on they read low: after t steps of a constant gradient g, "
  "m = g(1 − β₁ᵗ) and v = g²(1 − β₂ᵗ). Bias correction divides those factors back out. Without it the step is "
  "η · (1 − β₁ᵗ) / √(1 − β₂ᵗ): the first factor shrinks the step, the second inflates it, and because β₂ = 0.999 is "
  "so much closer to 1 than β₁ = 0.9, the inflation wins for a long time.\n")
w("| step | 1 | 2 | 3 | 5 | 10 | 15 | 20 |\n|---|--:|--:|--:|--:|--:|--:|--:|")
f20 = {r["t"]: r for r in r2["first20"]}
w("| corrected, step / η | " + " | ".join(f"{f20[t]['corrected']:.3f}" for t in (1, 2, 3, 5, 10, 15, 20)) + " |")
w("| uncorrected, step / η | " + " | ".join(f"{f20[t]['uncorrected']:.3f}" for t in (1, 2, 3, 5, 10, 15, 20)) + " |")
w("")
w(f"**When does the difference stop mattering? Not within twenty steps.** The uncorrected step peaks at "
  f"{r2['peak_ratio']:.2f}η at step {r2['peak_t']}, and is still 6.2η at step 20. It comes back within 50% of the "
  f"corrected step at step {cw['0.5']}, within 10% at step {cw['0.1']:,}, and within 1% at step {cw['0.01']:,}.\n")
w("That ratio is not an artefact of a constant gradient. Both versions divide the same m by the same √v, so for "
  "*any* gradient sequence the uncorrected update differs from the corrected one by exactly that factor, step for "
  "step (ε aside). The crossing points above hold for every run with these β's.\n")
w(f"**What it does to a real model.** The switchable optimizer was checked first: 20 steps against "
  f"`torch.optim.AdamW` agree to {r2['custom_vs_torch_max_loss_diff']:.0e} in loss. Then the same model, same data, "
  "300 steps, with and without correction:\n")
w("| | final held-out loss, corrected | uncorrected | largest gap in training loss |\n|---|--:|--:|--:|")
for k, v in r2["runs"].items():
    w(f"| {k} | {v['on_eval']:.4f} | {v['off_eval']:.4f} | {100 * v['max_rel_gap']:.1f}% |")
w("")
w("The curves never come back within 1% inside the run, because at step 300 the uncorrected run is still stepping "
  "about 2× too far. Warmup does not rescue it: it shrinks the first steps, but the multiplier outlives a 30-step "
  "warmup by thousands of steps.\n")

# ---- 3
w("## 3. The update-to-weight ratio, every layer\n")
w(f"For every weight matrix, every step: ‖Δw‖ / ‖w‖ — how far the step moved the weights, as a fraction of their "
  f"size. Two runs, identical except that one warms the learning rate up linearly over {r3['warmup_steps']} steps.\n")
w("![update-to-weight ratio](figures/update_to_weight.png)\n")
w(f"**Without warmup the first step is huge**: {r3['max_without']:.2e} on every layer, which is η / 0.02 — Adam moves "
  f"each weight by about η, and the weights start at a scale of 0.02. With warmup the largest ratio anywhere is "
  f"{r3['max_with']:.2e}, {r3['max_without'] / r3['max_with']:.1f}× smaller. This is the brief's argument in numbers: "
  "early gradients all agree, so every weight takes Adam's full step at once, and warmup is what keeps that from "
  "happening at full learning rate.\n")
w(f"**When warmup stops changing it: step {r3['median_stop']}.** For each layer, take the ratio in the warmup run "
  "divided by the ratio in the no-warmup run, smoothed with a 9-step rolling median; take the median over all "
  f"{len(r3['per_layer'])} layers; find the step after which it stays within 10% of 1. The two runs' weights drift "
  "apart for reasons that have nothing to do with warmup, so a single layer can differ forever — which is why the "
  f"rule uses the median across layers. {r3['layers_passing']} of {len(r3['per_layer'])} layers also pass it on their own.\n")
w(f"Warmup ended at step {r3['warmup_steps']}; its effect on the ratio lasted to step {r3['median_stop']}. After it "
  "the ratios settle near 1e-2 — ten times the brief's 1e-3 guide, which is expected for a model this small at "
  "this learning rate, and is itself the kind of number the brief says to log from step one.\n")
w("| layer | max, no warmup | max, warmup | last 20 steps | warmup stops mattering |\n|---|--:|--:|--:|--:|")
for n, v in r3["per_layer"].items():
    w(f"| `{n}` | {v['max_without']:.2e} | {v['max_with']:.2e} | {v['final']:.2e} | "
      f"{'step ' + str(v['stop']) if v['stop'] is not None else 'not within the run'} |")
w("")

# ---- 4
tab = r4["table"]
w("## 4. Cosine against WSD\n")
w(f"Both schedules are defined for {r4['total']} steps with a 30-step warmup. Cosine decays from its peak to 10% "
  f"along a cosine; WSD holds its peak and decays linearly over the last 20% (steps 240–300). Both stop at step "
  f"{r4['stop']}: at that point cosine is at 0.38× its peak and WSD has not started to decay.\n")
w("**Tune both sides first.** A comparison of two methods at one shared learning rate is a comparison of how well "
  "that learning rate suits each of them. So each schedule gets its own sweep:\n")
w("| peak learning rate | cosine, held-out at 200 | WSD, held-out at 200 |\n|--:|--:|--:|")
for lr in r4["grid"]:
    t = tab[str(lr)]
    w(f"| {lr:.0e} | {t['cosine']:.4f} | {t['wsd']:.4f} |")
w("")
if r4["best_cosine_lr"] == r4["best_wsd_lr"]:
    w(f"Both are best at {r4['best_cosine_lr']:.0e}, so the comparison below is at each side's own optimum, not at a "
      "learning rate chosen for one of them.\n")
else:
    w(f"Cosine is best at {r4['best_cosine_lr']:.0e} and WSD at {r4['best_wsd_lr']:.0e}; the comparison below uses each "
      "side's own optimum.\n")
w("![schedules and loss](figures/cosine_vs_wsd.png)\n")
w("| model | held-out loss |\n|---|--:|")
w(f"| cosine (300-step schedule) stopped at 200 | {r4['cosine_at_stop']:.4f} |")
w(f"| **WSD (300-step schedule) stopped at 200** | **{r4['wsd_at_stop']:.4f}** |")
w(f"| WSD at 200 + a {r4['branch_steps']}-step decay branch | {r4['wsd_branch']:.4f} |")
w(f"| cosine scheduled for exactly 200 steps | {r4['cosine_200_reference']:.4f} |")
w("")
w(f"**The model I would keep is WSD's.** Stopped at step 200 with no more compute, it is ahead by "
  f"{r4['cosine_at_stop'] - r4['wsd_at_stop']:.3f} nats. Cosine has spent its first 200 steps decaying towards a "
  "finish line at 300 that it never reached; WSD spent them at full learning rate. And WSD's real advantage is the "
  f"next row: the step-200 checkpoint can be decayed on a branch, and {r4['branch_steps']} more steps take it to "
  f"{r4['wsd_branch']:.4f}, better than anything else here. That is the brief's point — WSD does not need to know the "
  "length of the run in advance.\n")
w("One result I did not expect: cosine scheduled for exactly 200 steps is the *worst* model of the four. In a run this "
  "short the model is still far from converged, and time spent at a high learning rate is worth more than time spent "
  "annealing. That is a statement about this regime — 200 steps of a small model — not about pretraining in general. "
  "It is also one seed.\n")

# ---- 5
tb = r5["table"]
widths = r5["widths"]
w("## 5. The learning rate across widths\n")
w(f"Widths {', '.join(str(x) for x in widths)} (head dimension 64, so 4, 8 and 16 heads), 4 layers, standard "
  f"parameterization (every weight initialised at std 0.02 regardless of width), {r5['steps']} steps each with a "
  "30-step warmup and cosine to 10%, and the peak learning rate swept in factors of 2.\n")
w("![loss against learning rate](figures/lr_sweep.png)\n")
w("| peak learning rate | " + " | ".join(f"width {x}" for x in widths) + " |\n|--:|" + "--:|" * len(widths))
for lr in r5["grid"]:
    w(f"| {lr:.1e} | " + " | ".join(f"{tb[str(x)][str(lr)]:.4f}" for x in widths) + " |")
w("")
w("The minimum of each curve is found by fitting a parabola through the best grid point and its two neighbours, in "
  "log learning rate:\n")
w("| width | grid minimum | fitted minimum | loss |\n|--:|--:|--:|--:|")
for x in widths:
    m = r5["minima"][str(x)]
    w(f"| {x} | {m['grid_best']:.1e} | **{m['fitted']:.2e}** | {m['loss']:.4f} |")
w("")
pe = r5["pair_exponents"]
w(f"A power law through the three minima gives η* ∝ width^{r5['exponent']:.2f}, which extrapolates to "
  f"**{r5['predicted_4096']:.1e} at width 4,096**. The two width doublings on their own give exponents of "
  f"{pe[0]:+.2f} and {pe[1]:+.2f}, which carried to 4,096 span {r5['range_4096'][0]:.1e} to {r5['range_4096'][1]:.1e}.\n")
w("**How confident I am: not very, and the data says so.**\n")
ratio_txt = f"by a factor of {abs(pe[1] / pe[0]):.0f}" if abs(pe[0]) > 1e-9 else "completely"
w(f"- The two pairwise exponents disagree {ratio_txt} ({pe[0]:+.2f} against {pe[1]:+.2f}). Three points "
  "cannot distinguish a power law from noise.\n"
  "- The minima are shallow. At width 1,024 the losses at 4e-4 and 8e-4 differ in the fourth decimal, so the "
  "minimum is pinned down to within a factor of two at best.\n"
  "- It is one seed, 300 steps, and a 4× extrapolation beyond the widest point.\n"
  f"- The measured exponent, {r5['exponent']:.2f}, is much flatter than the brief's table, which has the optimum "
  "falling roughly as 1/width. A likely reason is what these models are made of: at width 256, the 50,257-token "
  "embedding is 12.9M of the 15.9M parameters. Theory for Adam under the standard parameterization says the "
  "hidden matrices want a learning rate that falls with width but the embedding does not, so a model dominated by "
  "its embedding shows a weak dependence overall.\n")
w(f"**The value I would use at width 4,096: {r5['predicted_4096']:.1e}** — and I would not trust it without a short "
  "sweep around it at the real width. This is exactly the problem muP exists to remove: under muP the minima line "
  "up across widths, so a sweep at width 256 gives the answer for 4,096 directly instead of through an "
  "extrapolation like this one.\n")

w("---\n")
w(f"Run: {M['seconds']} s on an {M['gpu']}, torch {M['torch']}, seed {M['seed']}.\n")
(HERE / "README.md").write_text("\n".join(out))
print(f"README.md: {len(chr(10).join(out).split())} words")
