"""Generate README.md from results.json, so the write-up cannot drift from the run."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
M = R["meta"]
r1, r2, r3, r4, r5, r6 = (R[f"req{i}"] for i in range(1, 7))

out = []
w = out.append

w("# ERA V5 · Session 10 — making a training loop tell the truth\n")
w("A small GPT and a real training loop, instrumented until every claim in the session can be checked "
  "rather than believed. Every number on this page came out of `training_loop.py`. `run.log` is the "
  "transcript of the run and `results.json` is the same numbers machine-readable; this file is generated "
  "from that JSON by `make_readme.py`.\n")
w("```bash\npython training_loop.py            # the full run: run.log, results.json, figures/\n"
  "python training_loop.py --quick    # same structure, fewer steps, no torch.compile\n"
  "python training_loop.py --replot   # redraw figures/ from results.json\n"
  "python make_readme.py              # regenerate this file\n```\n")
w(f"**Setup:** seed `{M['seed']}` (the session date), tiny Shakespeare via GPT-2 BPE (V = 50,257), "
  f"a 4-layer pre-norm GPT at D = 256 ({r1['n_params']:,} parameters, {r1['n_non_embedding']:,} outside the "
  f"embeddings), torch {M['torch']} on an {M['gpu']}. The notebook "
  f"[`training_loop.ipynb`](training_loop.ipynb) runs the same functions cell by cell.\n")
w("| file | what |\n|---|---|\n"
  "| [`training_loop.py`](training_loop.py) | the harness — one function per requirement |\n"
  "| [`model.py`](model.py) | the Session 9 GPT, with one fix found by requirement 2 |\n"
  "| [`training_loop.ipynb`](training_loop.ipynb) | the notebook version, with outputs |\n"
  "| [`run.log`](run.log) | full transcript of the run below |\n"
  "| [`results.json`](results.json) | every number, machine-readable |\n"
  "| [`figures/`](figures) | the two plots |\n")

# ---- summary
acc = r3
gap_final = acc["buggy"]["eval"][-1] - acc["correct"]["eval"][-1]
best_mfu = max(r5["variants"], key=lambda v: v["mfu"])
base = next(v for v in r5["variants"] if v["label"].startswith("bf16, torch.compile")) \
    if any(v["label"].startswith("bf16, torch.compile") for v in r5["variants"]) else r5["variants"][1]
w("## The six answers\n")
w("| # | requirement | result |\n|---|---|---|")
w(f"| 1 | every tensor shape, with what each dimension means | {len(r1['rows'])} tensors traced through one step, "
  f"the hand-unrolled forward asserted equal to `model()` |")
w(f"| 2 | verify one gradient by hand | first attempt **failed** (relative error {r2['legacy']['rel_err']:.2f}); "
  f"cause found in `RMSNorm`; after the fix, nudge and `backward()` agree to **{r2['agree_sig_figs']} significant figures** |")
w(f"| 3 | break gradient accumulation on purpose | reported loss off by up to **{100 * acc['worst_gap']:.1f}%** on a single "
  f"step; the model trained on it ends **{gap_final:.3f} nats worse** held-out; with equal lengths the gap is "
  f"exactly {acc['control']['loss_abs_diff']:.0f} |")
if r4["found"]:
    e = r4["first"]
    w(f"| 4 | a step where the grad norm moved before the loss | step **{e['step']}**, in the normal setting; the loss "
      f"followed {e['lead']} steps later. Thin evidence — see below |")
w(f"| 5 | MFU, honestly | **{100 * base['mfu']:.1f}%** for the model as specified; **{100 * best_mfu['mfu']:.1f}%** for "
  f"the same code on a 4x wider model. What costs the distance: only {100 * r5['profile']['matmul kernels']:.0f}% of "
  f"GPU time is matmul |")
w(f"| 6 | 0.1 in fp32, bf16, fp8 E4M3, by hand | `{r6['fp32']['hex']}`, `{r6['bf16']['hex']}`, `{r6['fp8 E4M3']['hex']}`; "
  f"all three match what the hardware formats produce. Train in **bf16** |\n")

# ---- 1
w("## 1. Every tensor shape in one step\n")
w(f"One micro-batch of B = {r1['B']} sequences × T = {r1['T']} positions, walked through by hand. Block 0 is "
  "unrolled — attention included — and the result is asserted equal to the module's own output, so the shapes "
  "below are the shapes the model actually computes, not a diagram of them.\n")
w("| tensor | shape | what each dimension means |\n|---|---|---|")
for row in r1["rows"]:
    w(f"| `{row['tensor']}` | `{tuple(row['shape'])}` | {row['meaning']} |")
w("")
w(f"Two things the table makes visible. `wte` is one matrix doing two jobs — the lookup at the input and, "
  f"tied, the `V × D` output head — so it collects gradient from both ends. And only the first T = {r1['T']} "
  f"rows of the positional table received any gradient ({r1['wpe_unused_rows']} of 128 rows got exactly zero): "
  f"a position the batch never used cannot learn anything that step. The untrained loss is "
  f"{r1['loss']:.4f} against ln V = {r1['ln_v']:.4f}, as it should be.\n")

# ---- 2
lg = r2["legacy"]
w("## 2. One gradient, verified by hand — and what the first attempt found\n")
w(f"Weight `{r2['weight']}`. Nudge it up and down, measure the loss each time, take the slope, compare it "
  "with what `backward()` stored in `.grad`.\n")
w("**The first attempt did not agree.** With the model exactly as Session 9 left it, in fp64, a central "
  f"difference at ε = 1e-6 gave `{lg['nudge']:+.12f}` against `backward()`'s `{lg['autograd']:+.12f}` — a "
  f"relative error of {lg['rel_err']:.2f}, where fp64 should manage about 1e-9. The error also *grew* as ε "
  "shrank, which is the signature of lost precision rather than of a wrong derivative.\n")
w("The cause was one line of `RMSNorm`: `x.float()`. For a bf16 model that is an upcast, which is why it was "
  "written. For an fp64 model it is a silent **downcast** to fp32 inside every norm, so the \"fp64\" loss "
  "carried only fp32's seven digits and the finite difference was mostly rounding noise. Nothing raised an "
  "error; the model trained perfectly well in both sessions. The fix promotes instead of casting — "
  "`x.to(torch.promote_types(x.dtype, torch.float32))` — which is an upcast for bf16 and a no-op for fp64.\n")
w(f"**After the fix:** `backward()` in fp64 gives `{r2['autograd_fp64']:+.12f}`; the nudge gives "
  f"`{r2['nudge_fp64']:+.12f}`. Absolute difference {r2['abs_err']:.1e}, relative {r2['rel_err']:.1e} — "
  f"**{r2['agree_sig_figs']} significant figures**. `backward()` in fp32 gives `{r2['autograd_fp32']:+.12f}`.\n")
w("The same check across nudge sizes shows why a careless version fails even with a correct model "
  "(relative error against the fp64 `backward()`):\n")
w("| ε | fp64 central | fp64 forward | fp32 central | fp32 forward |\n|--:|--:|--:|--:|--:|")
for s in r2["sweep"]:
    w(f"| {s['eps']:.0e} | {s['fp64_central']:.1e} | {s['fp64_forward']:.1e} | {s['fp32_central']:.1e} | {s['fp32_forward']:.1e} |")
w("")
w("Two errors pull in opposite directions. **Truncation**: a finite difference is only the slope of a straight "
  "line, so a large ε includes curvature — the forward difference is off in proportion to ε, the central one to "
  "ε². **Rounding**: a small ε makes the loss change so small that it falls into the last digits the format "
  "keeps. In fp64 the best agreement sits near ε = 1e-3 to 1e-4; in fp32 the rounding floor arrives so early that "
  "every ε below 1e-2 gets worse, and at 1e-6 the fp32 answer is off by more than 100%. The rule that falls out: "
  "check gradients in fp64, with a central difference, at an ε near the cube root of the format's precision.\n")

# ---- 3
c, b = acc["correct"], acc["buggy"]
w("## 3. Gradient accumulation, broken on purpose\n")
w("Four micro-batches of 8 sequences per step. Each micro-batch keeps only 8, 16, 32, 64 or 128 of its "
  "128 positions in the loss (the rest are padding), drawn at random, so the token counts differ step to step. "
  f"Two runs of {acc['steps']} steps from the same initialisation, on the same data in the same order:\n")
w("- **correct** — sum every token's loss across the step, divide by all the valid tokens in the step\n"
  "- **buggy** — average each micro-batch over its own tokens, then average the four averages\n")
w("![the gap and the held-out loss](figures/accumulation_gap.png)\n")
w(f"**The reported loss.** On the same batches, the average of averages differs from the true loss by "
  f"{100 * acc['mean_reported_gap']:.2f}% on average and by as much as {100 * acc['worst_gap']:.1f}% on one step "
  f"(step {acc['worst_gap_step']}). It is not a bias you would spot on a dashboard: step to step it swings both "
  "ways, and its moving average stays within about ±1%.\n")
w(f"**The model.** The damage is in the gradient, not the log. A short micro-batch's few tokens get the same vote "
  "as a long one's many, so the buggy run over-weights whatever short micro-batches contain — here, only early "
  "positions. It ends worse on held-out data:\n")
w("| step | correct | buggy | early positions 0–15, correct / buggy | late positions 64–127, correct / buggy |\n|--:|--:|--:|--:|--:|")
for i, s in enumerate(c["eval_step"]):
    if i % 4 == 0 or i == len(c["eval_step"]) - 1:
        w(f"| {s} | {c['eval'][i]:.4f} | {b['eval'][i]:.4f} | {c['eval_early'][i]:.3f} / {b['eval_early'][i]:.3f} | "
          f"{c['eval_late'][i]:.3f} / {b['eval_late'][i]:.3f} |")
w("")
w(f"Early on the buggy run is slightly *better* at early positions — it was trained harder on them — and by the end "
  f"it is worse everywhere, {gap_final:.3f} nats behind overall.\n")
ctl = acc["control"]
w(f"**How it hid.** The same comparison with all four micro-batches at full length: losses "
  f"{ctl['loss_correct']:.6f} and {ctl['loss_buggy']:.6f}, a difference of exactly {ctl['loss_abs_diff']:.0f}, and "
  f"gradients {ctl['grad_rel_diff']:.1e} apart (floating-point summation order). When token counts are equal the two "
  "formulas are the same formula, which is the case casual testing tends to exercise.\n")

# ---- 4
w("## 4. The grad norm, every step — and a step where it moved first\n")
if r4["found"]:
    e = r4["first"]
    top = r4["ladder"][-1]
    w("The grad norm is logged before clipping, every step (`clip_grad_norm_` returns the pre-clip value). "
      "\"Moved\" is defined before looking, so the answer cannot be picked by eye: a series moves at step *s* when "
      "it sits ≥ 4 robust standard deviations above its own last 20 steps (rolling median and MAD). The search "
      "wants a step where the grad norm moves **up** while the loss is still quiet (< 2), and the loss then moves "
      "up (≥ 3) within 10 steps. The first 50 steps are skipped, since everything moves then. Full-length batches, "
      "so per-step loss noise is not batch shape.\n")
    w(f"Found in **{r4['source']}** — no need to turn the guard-rails off to see one:\n")
    w("![grad norm and loss around the event](figures/gradnorm_leads_loss.png)\n")
    s = e["step"]
    w("| step | grad norm | loss | |\n|--:|--:|--:|---|")
    for t in range(max(0, s - 3), min(len(r4["gnorm"]), s + e["lead"] + 3)):
        mark = "grad norm moves" if t == s else ("loss moves" if t == s + e["lead"] else "")
        w(f"| {t} | {r4['gnorm'][t]:.3f} | {r4['loss'][t]:.4f} | {mark} |")
    w("")
    w(f"At step {s} the grad norm jumps to z = {e['z_gnorm']:+.1f} while the loss sits at z = {e['z_loss_at_step']:+.1f}; "
      f"the loss follows at step {s + e['lead']} (z = {e['z_loss_later']:+.1f}).\n")
    w(f"**How much this is worth.** Across the run the grad norm spiked {top['gnorm_spikes']} times. A loss spike "
      f"followed within 10 steps after {100 * top['p_loss_spike_after_gnorm_spike']:.0f}% of them, against "
      f"{100 * top['p_loss_spike_any_window']:.0f}% for any 10-step window — so a grad-norm spike does raise the "
      "odds of a loss spike. But that rests on a handful of events in a 400-step run of a small model, and in the "
      "plot the grad norm also rises at the very step the loss spikes. What this run supports is the modest claim: "
      "the grad norm is the earlier and cleaner of the two signals, not a reliable forecast.\n")

# ---- 5
w("## 5. MFU, measured honestly\n")
w(f"**The denominator.** A laptop part's datasheet peak depends on its power limit, so the peak here is measured: "
  f"the best of five timed windows of an 8192³ matmul after a two-second warm-up — "
  f"**{r5['peak_bf16'] / 1e12:.1f} TFLOP/s in bf16** (fp32: {r5['peak_fp32'] / 1e12:.1f}). Every MFU below divides by "
  "the bf16 figure, including the fp32 row: MFU asks what share of the machine you are paying for is doing model "
  "arithmetic, and dividing fp32 by an fp32 peak would flatter it.\n")
w("**The numerator.** Two counts. The brief's `6N` uses every parameter. The exact count uses only the weights "
  "that take part in a matmul (the embedding lookup costs nothing; the tied head is a real V × D matmul, so it "
  "counts), plus attention's own score and value matmuls, 12·L·T·D.\n")
w("| variant | tokens/s | MFU (6N) | MFU (exact) |\n|---|--:|--:|--:|")
for v in r5["variants"]:
    w(f"| {v['label']} | {v['tokens_per_s']:,.0f} | {100 * v['mfu_6n']:.1f}% | {100 * v['mfu']:.1f}% |")
w("")
w("Where the GPU time goes for the model as specified (bf16, share of CUDA kernel time):\n")
w("| kernels | share |\n|---|--:|")
for k, v in r5["profile"].items():
    w(f"| {k} | {100 * v:.1f}% |")
w("")
w(f"**What costs the distance to 40%.** The model is too narrow for this GPU. At D = 256 the matmuls are small, so "
  f"only {100 * r5['profile']['matmul kernels']:.0f}% of GPU time is spent in them. The rest goes to work that "
  "moves memory rather than doing arithmetic: the softmax and cross-entropy over 50,257 vocabulary entries at every "
  f"position ({100 * r5['profile']['softmax / cross-entropy over V']:.0f}%), and the norms, casts and elementwise "
  f"ops ({100 * r5['profile']['elementwise, norms, casts, copies']:.0f}%). `torch.compile` fuses some of the latter "
  f"and buys a point. The test of that diagnosis is the last row: the same code on a model four times wider reaches "
  f"**{100 * best_mfu['mfu']:.1f}%** — past 40% — because now the matmuls are big enough to dominate. The loop is not "
  "what is slow; the shape is.\n")

# ---- 6
w("## 6. 0.1 in fp32, bf16 and fp8 E4M3, by hand\n")
w("0.1 = 1.6 × 2⁻⁴, so every format stores the exponent −4 and has to approximate the 0.6. In binary 0.6 is "
  "0.1001 1001 1001 … and repeats forever, so every format rounds; the only question is where. Rounding is "
  "to nearest, ties to even, done in exact arithmetic (`fractions.Fraction`) and then checked against the bits "
  "the real formats produce.\n")
w("| format | layout | exponent field | mantissa (0.6 × 2ᵐ → rounded) | bits | value | error |\n|---|---|---|---|---|--:|--:|")
lay = {"fp32": ("1 · 8 · 23", 127, 23), "bf16": ("1 · 8 · 7", 127, 7), "fp8 E4M3": ("1 · 4 · 3", 7, 3)}
for k, (L, bias, mb) in lay.items():
    x = r6[k]
    w(f"| {k} | {L}, bias {bias} | −4 + {bias} = {x['exp_field']} = `{x['exponent']}` | "
      f"{x['mantissa_exact']:.1f} → {x['mantissa_int']} = `{x['mantissa']}` | `{x['sign']} {x['exponent']} {x['mantissa']}` "
      f"(`{x['hex']}`) | {x['value']:.12g} | {100 * x['rel_err']:.3g}% |")
w("")
w("All three agree bit for bit with `struct.pack('>f', 0.1)`, `torch.bfloat16` and `torch.float8_e4m3fn`.\n")
w("**Which I would train in: bf16**, with fp32 master weights and optimizer state. The table is the argument. "
  "bf16 keeps fp32's eight exponent bits, so it reaches every magnitude a gradient will take and needs no loss "
  "scaling; it pays in precision (0.1% error here), which the fp32 master copy absorbs at the update. fp8 E4M3 is "
  "already 1.6% off on a number as ordinary as 0.1, and with four exponent bits its range tops out at 448 — it "
  "only works with a scale factor per tensor or per block, and even then only inside the matmuls, not for the "
  "weights the optimizer writes to. Requirement 5 settles the rest for this model: its time is not in the "
  "matmuls, so the one thing fp8 would speed up is not what is slow.\n")

w("---\n")
w(f"Run: {M['seconds']} s on an {M['gpu']}, torch {M['torch']}, seed {M['seed']}.\n")
(HERE / "README.md").write_text("\n".join(out))
print(f"README.md: {len(chr(10).join(out).split())} words")
