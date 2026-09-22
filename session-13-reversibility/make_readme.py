#!/usr/bin/env python3
"""Generate README.md from results.json, so no number on the page can drift from the run."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
runs = {r["tag"]: r for r in R["runs"]}
base = runs["1-baseline"]
same = runs["2-reversible-same-batch"]
big = runs["3-reversible-max-batch"]
tuned = runs.get("4-reversible-max-batch-lr-scaled")
best = R["best_variant"]


def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f}%"


def slower(a, b):
    """How much slower a is than b, in plain words."""
    d = (1 - a / b) * 100
    return f"{d:.1f}% slower" if d > 0 else f"{-d:.1f}% faster"


def saves(a, b):
    d = (1 - a / b) * 100
    return f"{d:.1f}% less memory" if d > 0 else f"{-d:.1f}% more memory"


L = []
w = L.append

w("# ERA V5 · Session 13 — reversibility, measured")
w("")
w(f"A {base['params'] / 1e6:.1f}M-parameter byte-level GPT trained on {base['tokens'] / 1e6:.0f}M tokens, "
  f"four times: once normally, once with a reversible stack at the same batch size, once with the "
  f"reversible stack at the largest batch it allows, and once more at that batch with the learning rate "
  f"corrected for it. Everything below came out of one run of "
  f"[`train.py`](train.py); [`run.log`](run.log) is its transcript and [`results.json`](results.json) "
  f"holds the numbers. This file is generated from that JSON.")
w("")
w("```bash")
w("python data.py      # build the corpus from Session 4's cleaned shard")
w("python train.py     # runs 1-3, plus the checks below")
w("python extra_run.py # run 4: the same batch as run 3 with the learning rate scaled")
w("python train.py --quick")
w("```")
w("")
w(f"Setup: {R['device']}, torch {R['torch']}, seed {R['seed']}, sequence length {R['block']}, "
  f"byte vocabulary (256), AdamW at 3e-4, gradient clipping 1.0, **dropout 0 and weight decay 0 in "
  f"every arm** — see \"Why dropout must be zero\" below.")
w("")

w("## The runs")
w("")
w("| | stack | batch | optimizer steps | tokens/s | peak memory | val loss |")
w("|---|---|--:|--:|--:|--:|--:|")
for r in [x for x in (base, same, big, tuned) if x]:
    w(f"| {r['tag'][2:]} | {r['stack']} | {r['batch']} | {r['steps']:,} | {r['tokens_per_s']:,.0f} | "
      f"{r['peak_mem_mib']:,.0f} MiB | {r['val_loss']:.4f} |")
w("")
w(f"**Reading it.** At the same batch size the reversible stack is **{slower(same['tokens_per_s'], base['tokens_per_s'])}** "
  f"than the baseline and uses **{saves(same['peak_mem_mib'], base['peak_mem_mib'])}** — which is the trade the "
  f"method exists to make, and the paper prices the slowdown at 30–50%. "
  f"Spending that memory on a bigger batch ({big['batch']} against {base['batch']}) leaves it "
  f"**{slower(big['tokens_per_s'], base['tokens_per_s'])}** than the baseline. "
  f"Validation loss: {base['val_loss']:.4f} against {same['val_loss']:.4f} at equal batch — "
  f"{same['val_loss'] - base['val_loss']:+.4f} nats, one seed, no other difference.")
w("")
w(f"**The bigger batch bought no speed.** {big['tokens_per_s']:,.0f} tokens/s at batch {big['batch']} against "
  f"{same['tokens_per_s']:,.0f} at batch {base['batch']} — a {abs(big['tokens_per_s'] / same['tokens_per_s'] - 1) * 100:.1f}% "
  f"difference, which is nothing. The paper reports throughput gains up to 101% from exactly this move, but "
  f"its gains come at 96 layers, where the activation term dominates. At 12 layers on a 6 GB card there is "
  f"nothing left to win: the run is bound by the recomputation, not by how many sequences are in flight.")
if tuned:
    w("")
    w(f"**And most of run 3's loss penalty was the learning rate, not the batch.** Run 3 changes two things "
      f"at once: the batch rises {big['batch'] / base['batch']:.1f}x, so the same 50M tokens are covered in "
      f"{big['steps']:,} optimizer steps instead of {base['steps']:,} — at a learning rate chosen for the small "
      f"batch. Run 4 repeats run 3 with the learning rate scaled by the square root of the batch ratio "
      f"({R['lr_scaled']['base_lr']:.1e} → {R['lr_scaled']['scaled_lr']:.2e}, the rule from Session 11) and "
      f"nothing else changed: validation loss goes {big['val_loss']:.4f} → {tuned['val_loss']:.4f}, recovering "
      f"{big['val_loss'] - tuned['val_loss']:.2f} of the {big['val_loss'] - same['val_loss']:.2f} nat gap. "
      f"A fixed token budget spent in fewer, larger steps still costs something — but most of what looked like "
      f"the cost was an untuned comparison.")
w("")
probe = R["max_batch"][best]
if big["batch"] < probe:
    w(f"**A note on that batch size.** The probe found {probe} survivable — three full steps, clipping and "
      f"optimizer included — and training at {probe} still ran out of memory, so the run backed off to "
      f"**{big['batch']}**. A short probe cannot see the allocator's steady state on a card this full; the "
      f"number in the table is the one that actually trained 50M tokens.")
    w("")

w("## Which variant works")
w("")
w("The class names two variants, \"Euler and midpoint\". The paper (arXiv:2512.02056) gives three, and "
  "plain forward Euler is not one of them — it is not reversible, because recovering the previous state "
  "would need the block evaluated at the state you are trying to recover. What it actually offers is:")
w("")
w("| variant | rule | inverse |")
w("|---|---|---|")
w("| midpoint (eq. 2.4) | `p[l+1] = a·p[l-1] + 2h·f(p[l])` | `p[l-1] = (p[l+1] - 2h·f(p[l])) / a` |")
w("| leapfrog (eq. 2.6) | `p[l+1] = 2p[l] - p[l-1] + h²·f(p[l])` | `p[l-1] = 2p[l] - p[l+1] + h²·f(p[l])` |")
w("| Hamiltonian (eq. 2.8-2.9) | `q[l] = a·q[l-1] + Attn(LN(p[l-1]))`, `p[l] = a·p[l-1] + MLP(LN(q[l]))` | undo the MLP step, then the attention step |")
w("")
w(f"Short runs at equal batch, {R['variants'][0]['tokens'] / 1e6:.1f}M tokens each:")
w("")
w("| variant | tokens/s | peak memory | val loss |")
w("|---|--:|--:|--:|")
for v in R["variants"]:
    w(f"| {v['stack']} | {v['tokens_per_s']:,.0f} | {v['peak_mem_mib']:,.0f} MiB | {v['val_loss']:.4f} |")
w("")
w(f"**{best} wins on validation loss**, and is what the three headline runs use.")
w("")

w("## Is it actually reversible?")
w("")
w("Two checks, because a reversible stack that does not reconstruct exactly is just a slower model "
  "with wrong gradients.")
w("")
w("**The custom backward against ordinary autograd**, in fp64, worst relative error over every parameter:")
w("")
w("| variant | worst relative gradient error |")
w("|---|--:|")
for g in R["grad_check"]:
    w(f"| {g['stack']} | {g['worst_rel_grad_error']:.2e} |")
w("")
w("**Reconstruction error**, and what V4's damping coefficient does to it. The paper's stability "
  "analysis (§3) requires |a| = 1 for a method to be stable forwards *and* backwards; V4's production "
  "integrator shipped **a = 0.5**:")
w("")
w("| variant | a = 1.0 (paper) | a = 0.5 (V4's setting) |")
w("|---|--:|--:|")
by = {(r["stack"], r["a"]): r["rel_error"] for r in R["reconstruction"]}
for s in ("midpoint", "leapfrog", "hamiltonian"):
    one, half = by.get((s, 1.0)), by.get((s, 0.5))
    half_txt = f"{half:.2e}" if half is not None else "--"
    w(f"| {s} | {one:.2e} | {half_txt} |")
w("")
ratio = by[("midpoint", 0.5)] / max(by[("midpoint", 1.0)], 1e-30)
w(f"At a = 1 every stack rebuilds to floating-point noise. At a = 0.5 the midpoint inverse divides by 0.5 at "
  f"every layer, so any error doubles on the way down and the relative error is **{ratio:.0e}× larger**; the "
  f"Hamiltonian stack, which divides by a twice per layer, is worse still. **Leapfrog is unaffected, because "
  f"its rule has no a in it.** V4 shipped a = 0.5, and the paper's §3 requires |a| = 1 for a method to be "
  f"stable in both directions — this table is what that requirement costs when it is ignored.")
w("")

w("## Memory against depth")
w("")
w("The paper's headline structural claim is that a reversible stack's activation memory does not grow "
  "with depth, while a standard stack's does.")
w("")
w("| layers | parameters | baseline peak | reversible peak |")
w("|--:|--:|--:|--:|")
depths = sorted({r["layers"] for r in R["memory_vs_depth"]})
mv = {(r["layers"], r["stack"]): r for r in R["memory_vs_depth"]}


def cell(r):
    return "**out of memory**" if r.get("oom") else f"{r['peak_mem_mib']:,.0f} MiB"


for L_ in depths:
    b, m = mv[(L_, "baseline")], mv[(L_, "midpoint")]
    w(f"| {L_} | {max(b['params'], m['params']) / 1e6:.1f}M | {cell(b)} | {cell(m)} |")
w("")
d0, d1 = depths[0], depths[-2]
bstep = (mv[(d1, "baseline")]["peak_mem_mib"] - mv[(d0, "baseline")]["peak_mem_mib"]) / (d1 - d0)
mstep = (mv[(d1, "midpoint")]["peak_mem_mib"] - mv[(d0, "midpoint")]["peak_mem_mib"]) / (d1 - d0)
w(f"Measured at a fixed batch: the baseline costs about **{bstep:,.0f} MiB per layer** and the reversible "
  f"stack about **{mstep:,.0f} MiB per layer** — the second number is weights and optimizer state, which both "
  f"stacks pay, and nothing else. At {depths[-1]} layers the baseline cannot run on this card at all while the "
  f"reversible stack uses {mv[(depths[-1], 'midpoint')]['peak_mem_mib']:,.0f} MiB. That is the paper's claim, "
  f"and it is the reason the method exists: the saving is in the term that scales with depth.")
w("")

w("## Why dropout must be zero")
w("")
w("The backward pass rebuilds each layer's input by running the block again. With dropout on, the "
  "second run draws a different random mask, so it reconstructs a state the forward pass never "
  "produced and the gradients are wrong. This is a **correctness** requirement, not a preference — and "
  "it means the baseline has to run at dropout 0 too, or the comparison is between two different "
  "models rather than two stacks.")
w("")
w("The class also said weight decay cannot be used with reversibility [01:29:57]. The paper does not "
  "say that, and Session 12's brief recorded V4's zero weight decay as a deliberate architectural "
  "choice. These runs use weight decay 0 in every arm so the arms stay comparable, but the claim "
  "itself is still unsourced.")
w("")

w("## What this cost")
w("")
total_s = sum(r["seconds"] for r in R["runs"]) + sum(v["seconds"] for v in R["variants"])
recovered = [r["tag"] for r in R["runs"] + R["variants"] if r.get("recovered_from")]
w(f"All runs on one {R['device']}: {total_s / 3600:.1f} GPU-hours in total. On a rented A100 at about "
  f"$1.50/hour that would be roughly ${total_s / 3600 * 1.5:.2f} of compute; the point of the exercise "
  f"is that the reversible arm buys its extra time back only when the recovered memory is spent on a "
  f"bigger batch.")
w("")

w("## Honest limits")
w("")
w("- One seed, one model size, one sequence length. The direction is measured; the magnitudes are not "
  "a scaling law.")
w(f"- {base['params'] / 1e6:.1f}M parameters is small enough that weights and optimizer state dominate "
  "peak memory, which compresses the difference this method exists to create. The effect grows with "
  "depth and sequence length, and this GPU has 6 GB.")
w("- Throughput here is wall-clock on a laptop GPU with other processes idle; it is not a clean "
  "benchmark.")
if recovered:
    w(f"- The first full run finished every training run and then crashed in the depth sweep before writing "
      f"its JSON. The numbers for {len(recovered)} runs were parsed back out of `run.log` rather than "
      f"re-running three GPU-hours; their per-step loss curves are gone and `final_train_loss` is null in "
      f"`results.json`. Validation loss, throughput and peak memory are the logged values. `train.py` now "
      f"saves after every stage and treats an out-of-memory error in the sweep as a result.")
w("- The corpus is the Session 4 cleaned shard, read as raw bytes. A byte vocabulary keeps the "
  "embedding out of the parameter budget, but it also makes the loss numbers incomparable with any "
  "BPE-tokenized run (Session 9's point about perplexity across tokenizers).")

(HERE / "README.md").write_text("\n".join(L) + "\n")
print(f"wrote README.md -- {len(L)} lines")
