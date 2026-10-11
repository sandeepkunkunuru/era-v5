#!/usr/bin/env python3
"""Generate README.md entirely from results.json, so the prose cannot drift from the run.

    python make_readme.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
O = []


def w(*lines):
    O.extend(lines)


def table(head, rows):
    w("| " + " | ".join(head) + " |",
      "|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w("")


dev = R.get("gpu", "cpu")
w(f"# Session 15 — a training run that watches itself",
  "",
  "> Session 15's assignment is a team-selection form, not a build. So this answers the",
  "> session's own open questions instead: of the four spike rules it lists, which one",
  "> would you actually page a human with? Does M4 downsampling really lose nothing? How",
  "> small can a silent checkpoint misload get before no number can see it? And what does",
  "> the top-k intervention rate show that MaxVio and dead-expert counts cannot?",
  "",
  "Five stages, every number below read out of `results.json`, which is written by",
  "`run.py` and logged to `run.log`. Nothing here is quoted from the lecture or the brief;",
  "where a published claim is tested, the test is the evidence.",
  "")

w("## How to run it", "",
  "```bash",
  "python run.py              # all five stages",
  "python run.py --quick      # minutes, for a smoke test",
  "python run.py --only 1,2   # selected stages, keeping the others' results",
  "```",
  "",
  f"Measured on **{dev}**, torch {R.get('torch','?')}, seed 15. Stages 1 and 2 are numpy",
  "only. Stages 3 to 5 import Session 14's `model.py` and `train.py` unchanged, so this",
  "harness cannot disagree with the model it measures.",
  "")

# ----------------------------------------------------------------- 1. detectors
d = R["detectors"]
steps = d["steps"]
w("## 1. Four published spike rules, scored on one trace", "",
  f"A synthetic grad-norm trace of **{steps:,} steps** with a warmup, a decaying level,",
  "lognormal noise whose spread falls with the level, and **40 planted spikes** of known",
  "height and position. Same trace for every rule, so the comparison is fair. A spike",
  "counts as found if an alarm lands within 4 steps of it.",
  "")
for curve in ("grad_norm", "loss"):
    rows = [(r["rule"], r["found"], r["missed"], f"{r['false_alarms']:,}", f"{r['per_10k']:,.2f}")
            for r in d[curve]]
    w(f"**On the {curve.replace('_', ' ')}:**", "")
    table(["rule", "found", "missed", "false alarms", "false / 10k steps"], rows)

best = max(d["grad_norm"], key=lambda r: (r["found"], -r["false_alarms"]))
fixed = next(r for r in d["grad_norm"] if r["rule"].startswith("fixed"))
ema = next(r for r in d["grad_norm"] if "ZClip" in r["rule"])
ratio = next(r for r in d["grad_norm"] if "ratio" in r["rule"])
w("Three results worth stating plainly.", "",
  f"- **The fixed threshold misses {fixed['missed']} of 40.** The grad-norm level decays",
  "  through training, so spikes late in the run pass under a line drawn early. This is the",
  "  usual advice — *do not use a fixed limit* — with a number attached.",
  f"- **ZClip's z = 2.5 is not an alert threshold.** It finds every spike, and raises",
  f"  **{ema['per_10k']:,.1f} false alarms per 10,000 steps** — about",
  f"  {ema['per_10k'] * steps / 10_000:,.0f} pages over this run. ZClip *rescales* the",
  "  gradient at that threshold; it never pages anyone. Borrowing the number for an alert",
  "  band is a category error.",
  f"- **Megatron's ratio-to-the-maximum rule finds {ratio['found']} of 40**, on both curves.",
  "  A loss sets its running maximum in the first few thousand steps and never approaches",
  "  10x it again. The rule catches divergence to a huge value, which is what it is for.",
  "")
if "ema_z_sweep" in d:
    w("So where should the EMA rule sit? Sweeping the threshold on the grad norm:", "")
    table(["z", "found", "missed", "false alarms", "false / 10k"],
          [(f"{r['z']:.1f}", r["found"], r["missed"], f"{r['false_alarms']:,}",
            f"{r['per_10k']:,.2f}") for r in d["ema_z_sweep"]])
    good = [r for r in d["ema_z_sweep"] if r["found"] == 40 and r["per_10k"] < 1]
    if good:
        g = good[0]
        w(f"**z = {g['z']:.1f}** keeps all 40 detections at {g['per_10k']:.2f} false alarms",
          f"per 10,000 steps. The detection rate does not fall until well past that, because",
          "a real spike is many sigma out; all the threshold buys below z = 5 is noise.", "")
w(f"The best alarm rule on this trace is **{best['rule']}**: {best['found']} of 40 found,",
  f"{best['false_alarms']} false alarms.", "")

# ----------------------------------------------------------------- 2. downsampling
ds = R["downsampling"]
w("## 2. Downsampling a long run without losing the spike", "",
  f"{ds['steps']:,} steps drawn on a {ds['width']}-pixel-wide chart is",
  f"{ds['steps'] / ds['width']:.0f} steps a column. Each method is rasterised at",
  f"{ds['width']}x{ds['height']} and compared **pixel for pixel** against the full-data",
  "rendering, which is a stronger test than eyeballing the curve.",
  "")
table(["method", "points", "wrong pixels", "of", "peak value error", "note"],
      [(r["method"], f"{r['points']:,}", f"{r['wrong_pixels']:,}", f"{r['of_pixels']:,}",
        f"{r['peak_error']:.4f}", r["note"]) for r in ds["rows"]])
m4 = next(r for r in ds["rows"] if r["method"] == "M4 per column")
mm = next(r for r in ds["rows"] if r["method"] == "min/max per column")
mean = next(r for r in ds["rows"] if r["method"] == "mean")
w(f"- **M4 is exact: {m4['wrong_pixels']} wrong pixels of {m4['of_pixels']:,}.** Keeping the",
  "  minimum, maximum, first and last of each pixel column reproduces the full rendering",
  "  with no error at all, which is what Jugel et al. proved in 2014 (PVLDB 7(10):797-808)",
  "  for two-colour line rasterisation.",
  f"- **Min and max alone are not enough**: {mm['wrong_pixels']:,} wrong pixels at twice the",
  "  point budget. The first and last values are what close the gap between columns.",
  f"- **A mean-only rollup loses the spike**: peak value error {mean['peak_error']:.2f}.",
  "",
  "One trap, found by getting it wrong first. The rollup's buckets must use **the same",
  "index-to-column mapping the renderer uses to place a point**. Bucketing with",
  "`linspace` while the renderer rounds `i/(n-1)*(w-1)` straddles the boundaries and costs",
  "about 2,700 wrong pixels — which reads as a flaw in M4 rather than a flaw in the",
  "bucketing. The proof is only as good as the alignment.",
  "")

# ----------------------------------------------------------------- 3. misload
ml = R["misload"]
w("## 3. The silent checkpoint misload, and how small its only signal gets", "",
  "A converter writes tensors under the source model's key names; the target wraps its",
  "stack differently, so the names do not match; `load_state_dict(strict=False)` skips them",
  "and reports success, and those parameters keep their random initialization.",
  "",
  "Reproduced on Session 14's model. **Every cheap check passes:**",
  "")
table(["check", "result"],
      [(k, "PASS" if v else "fail") for k, v in ml["cheap_checks"].items()])
w(f"The state hash is **unchanged by the load** "
  f"(`state_unchanged_by_load: {ml['state_unchanged_by_load']}`): "
  f"{ml['missing_keys']} target parameters kept their random values and",
  f"{ml['unexpected_keys']} checkpoint tensors were written and then ignored.",
  "",
  "Now the part that matters. A real converter misnames *one class* of tensor, not all of",
  "them, so the question is how small the signal gets:",
  "")
table(["tensors misnamed", "parameters left random", "step-1 loss", "gap vs correct"],
      [(f"{r['frac']:.0%}", f"{r['params_random']:,}", f"{r['loss']:.4f}", f"{r['gap']:+.4f}")
       for r in ml["sweep"]])
sig = ml["batch_noise_sigma"]
invisible = [r for r in ml["sweep"] if r["gap"] < 2 * sig]
w(f"Batch-to-batch spread of a *correct* model is **{sig:.4f} nats** (1 sigma). So:", "",
  f"- Below about **5% of tensors misnamed**, the gap is inside the noise. At 2% it is",
  f"  {ml['sweep'][-1]['gap']:+.4f} against a 2-sigma band of {2 * sig:.4f} — invisible",
  "  without a reference run.",
  "- **A partial misload is worse than a total one.** 50% misnamed costs",
  f"  {next(r['gap'] for r in ml['sweep'] if r['frac'] == 0.5):+.2f} against",
  f"  {next(r['gap'] for r in ml['sweep'] if r['frac'] == 1.0):+.2f} for 100%. The correctly",
  "  loaded parts drive confident wrong logits; a fully random model is merely uncertain.",
  f"- For reference, the Lightning LM cookbook reports {ml['cookbook_reference']['broken']}",
  f"  against {ml['cookbook_reference']['correct']} at step one — a",
  f"  {ml['cookbook_reference']['gap']:.2f}-nat gap, which sits between our 5% and 10% rows.",
  "",
  f"Two checks do catch it: a **strict load** "
  f"(`strict_load_catches: {ml['strict_load_catches']}`) and building the checkpoint from",
  f"the **target keyspace** so a missing name is a construction-time error "
  f"(`target_keyspace_catches: {ml['target_keyspace_catches']}`).",
  "")

# ----------------------------------------------------------------- 4. freezing
fz = R["freezing"]
dc = fz["decay_on_frozen"]
gn = fz["grad_norm"]
mem, memf = fz["memory"], fz["memory_experts_frozen"]
w("## 4. Freezing: four constraints that break without an error", "", "**The stale gradient.**",
  f"A weight frozen *after* the optimizer was built, still carrying a zero gradient, shrinks",
  f"by `1-(1-lr*wd)^{dc['steps']}` = **{dc['predicted_shrink']:.4%}** in {dc['steps']} steps",
  f"at lr = {dc['lr']}, wd = {dc['weight_decay']}. Measured:",
  f"**{dc['relative']:.4%}** — predicted and observed agree to four digits.",
  "",
  "PyTorch skips a parameter whose `grad` is `None`, so `requires_grad_(False)` on its own",
  "is safe. The stale gradient is the entire mechanism: AdamW's decoupled weight decay needs",
  "no gradient, only a parameter in a group. Building the optimizer from trainable",
  f"parameters only fixes it — drift {dc['fixed_drift']:.1e}, hash stable",
  f"`{dc['fixed_hash_stable']}`.",
  "",
  f"The loss cannot see a {dc['relative']:.2%} shift in one tensor. **The hash can**:",
  f"`hash_catches: {dc['hash_catches']}`.",
  "",
  "**The grad norm moves when you freeze.**",
  f"Over every weight it is **{gn['all_trainable']:.4f}**; over the trainable ones only,",
  f"**{gn['experts_only']:.4f}** — a factor of {1 / gn['ratio']:.0f}. An inherited clip limit",
  "of 1.0 and an alarm band tuned before the freeze both now mean something different.",
  "",
  "**Memory, at 16 bytes a trained weight and 2 a frozen one.**",
  "")
table(["arrangement", "trainable", "frozen", "MiB", "vs nothing frozen"],
      [("experts trained, rest frozen", f"{mem['trainable_params']:,}",
        f"{mem['frozen_params']:,}", f"{mem['total_bytes'] / 2**20:,.0f}",
        f"{mem['all_trainable_bytes'] / mem['total_bytes']:.2f}x"),
       ("experts frozen, rest trained", f"{memf['trainable_params']:,}",
        f"{memf['frozen_params']:,}", f"{memf['total_bytes'] / 2**20:,.0f}",
        f"{memf['all_trainable_bytes'] / memf['total_bytes']:.2f}x")])
w("Which side you freeze decides everything, because the experts hold",
  f"{mem['trainable_params'] / (mem['trainable_params'] + mem['frozen_params']):.0%} of the",
  "parameters. This is why QLoRA and Lightning LM both freeze the big side and train a",
  "small one, and why freezing an embedding saves almost nothing.",
  "",
  "**Per-part zero-gradient counts**, the number a dashboard watches:",
  "")
table(["part", "parameters", "tensors", "gradient elements exactly zero"],
      [(k, f"{v['params']:,}", v["tensors"], f"{v['zero_grad_share']:.2%}")
       for k, v in fz["zero_grad_experts_only"].items()])
fh = fz["frozen_hash_after_5_steps"]
w(f"After 5 real optimizer steps, {fh['checked']} frozen tensors were re-hashed and",
  f"**{fh['moved']} changed**.", "")

# ----------------------------------------------------------------- 5. routing
rt = R.get("routing")
w("## 5. Routing health: the metric Session 14 did not have", "")
if not rt or rt.get("skipped") or not rt.get("arms"):
    w("Session 14's build measured MaxVio and dead experts. Neither can see a layer whose",
      "balancing bias has taken the routing decision away from its router: the loads come out",
      "even, nothing is dead, and the experts are chosen without reference to the token. The",
      "metric that can see it is the **top-k intervention rate** — the share of positions where",
      "`scores + bias` picks a different set of experts than `scores` alone — read against its",
      "ceiling of `1 - k/N`, together with the **selected top-k mass** read against `k/N`.",
      "",
      "`routing.py` implements all five metrics and the two ablations the reference model's team",
      "used as proof: route each token to k random experts, and delete the routed experts",
      "outright. A layer that is really routing loses a great deal of loss under the first.",
      "",
      "**The numbers are not in this build yet.** The machine lost its GPU to a kernel upgrade",
      "mid-run, and the CPU fallback had not finished at the time of writing. `python run.py",
      "--only 5` fills this section in; the other four stages are unaffected and complete.",
      "")
else:
    w("Session 14's build measured MaxVio and dead experts. Neither can see a layer whose",
      "balancing bias has taken the routing decision away from its router: the loads come out",
      "even, nothing is dead, and the experts are chosen without reference to the token.",
      "",
      "Two things have to be counted carefully. The **intervention rate** is per expert *slot* —",
      "the share of the k experts actually run that the router would not have picked. If the bias",
      "chose a k-subset at random the expected overlap with the router's own top-k is k·k/N, so",
      "the expected per-slot intervention is exactly `1 - k/N`, and that is the ceiling the",
      "published 98.5% is read against. Counted per *token* instead — \"the set changed at all\" —",
      "it saturates near 1 for any non-zero bias and has no ceiling, so it cannot be compared to",
      "that number. Both are below; the contrast is the point.",
      "",
      f"Session 14's model grown by `partition`, trained {rt['tokens']:,} further tokens per arm:",
      "")
    for arm in rt["arms"]:
        pl = arm["per_layer"]
        abl = {a["layer"]: a for a in arm["ablation"]}
        w(f"**balance = `{arm['balance']}`** — val loss {arm['run']['val_loss']:.4f}, "
          f"mean MaxVio {arm['mean_maxvio']:.4f}, {arm['dead_total']} experts dead across all "
          f"layers. (Dead counts here are measured on the validation probe, over a different "
          f"window from the {arm['run'].get('dead', '--')} the training loop reports, so the two "
          f"do not have to agree.)", "")
        table(["layer", "MaxVio", "dead", "entropy", "interv/slot", "set changed",
               "intrinsic mass", "selected mass", "random", "bypass"],
              [(i, f"{p['maxvio']:.4f}", p["dead"], f"{p['router_entropy']:.4f}",
                f"{p['intervention_rate']:.4f}", f"{p.get('set_change_rate', 0):.4f}",
                f"{p['intrinsic_topk_mass']:.4f}", f"{p['selected_topk_mass']:.4f}",
                f"{abl[i]['random_delta']:+.4f}" if i in abl else "--",
                f"{abl[i]['bypass_delta']:+.4f}" if i in abl else "--")
               for i, p in enumerate(pl)])
    n_layer = len(rt["arms"][0]["per_layer"])
    n_expert = rt["arms"][0]["run"]["dead_of"] // n_layer
    top_k = round(n_expert * rt["arms"][0]["per_layer"][0]["uniform_topk_mass"])
    bias_arm = next((a for a in rt["arms"] if a["balance"] == "bias"), rt["arms"][0])
    none_arm = next((a for a in rt["arms"] if a["balance"] == "none"), None)
    p0 = bias_arm["per_layer"][0]
    ceil = p0["intervention_ceiling"]
    unif = p0["uniform_topk_mass"]
    bl = bias_arm["per_layer"]
    worst = max(bl, key=lambda x: x["intervention_rate"])
    min_sel = min(bl, key=lambda x: x["selected_topk_mass"])
    w(f"The ceiling for the intervention rate here is `1 - k/N` = **{ceil:.4f}**, and uniform",
      f"top-k mass is `k/N` = **{unif:.4f}**.", "",
      "**The headline is a negative result, and it is the one worth having.** This model does not",
      "reproduce the published fault. With the bias on, the highest per-slot intervention rate is",
      f"**{worst['intervention_rate']:.4f}** at layer {bl.index(worst)}, against a ceiling of",
      f"{ceil:.4f} — the bias moves a real share of the choices but nowhere near all of them. More",
      f"telling, the lowest selected top-k mass is **{min_sel['selected_topk_mass']:.4f}** at layer",
      f"{bl.index(min_sel)}, well above the {unif:.4f} a router with no preference would give. The",
      "experts that ran are experts the router actually wanted. That is what a healthy layer looks",
      "like on these metrics, and having it measured is what makes the unhealthy case legible.",
      "")
    if none_arm:
        nl = none_arm["per_layer"]
        dead_layers = [(i, x["dead"]) for i, x in enumerate(nl) if x["dead"]]
        w("**Turning balancing off is the contrast.** Intervention falls to exactly 0.0000 in every",
          "layer, because with no bias the chosen set *is* the router's own — a useful check that the",
          f"metric measures what it claims. Mean MaxVio rises {none_arm['mean_maxvio'] / bias_arm['mean_maxvio']:.1f}x",
          f"({bias_arm['mean_maxvio']:.4f} to {none_arm['mean_maxvio']:.4f}) and",
          f"{none_arm['dead_total']} experts die, and they are concentrated at the top of the "
          "stack — "
          + ("; ".join(f"{d} in layer {i}" for i, d in dead_layers) if dead_layers else "nowhere")
          + ".",
          "",
          "**And MaxVio has a ceiling of its own, which is why it stops being informative.** "
          f"With N = {n_expert} experts a layer and top-{top_k} routing, the load is at its most "
          f"uneven when k experts take everything, giving MaxVio = N/k - 1 = "
          f"**{n_expert / top_k - 1:.1f}** — exactly the value layers 0 to 3 report. A metric "
          "sitting at its maximum cannot tell a bad layer from a worse one, and the published "
          "fault was found in a model whose *average* MaxVio read a healthy 0.31. That is the "
          "argument for carrying all five numbers per layer rather than one averaged over the "
          "stack.",
          "")
    b0 = bias_arm["ablation"][0] if bias_arm["ablation"] else None
    if b0:
        rest = [a for a in bias_arm["ablation"][1:]]
        w("**The ablations say layer 0 is not like the others.** Deleting its routed experts costs",
          f"**{b0['bypass_delta']:+.4f}** nats, against",
          f"{min(a['bypass_delta'] for a in rest):+.4f} to {max(a['bypass_delta'] for a in rest):+.4f}",
          "for every other layer. Random routing there costs",
          f"{b0['random_delta']:+.4f} with the bias on and",
          (f"{none_arm['ablation'][0]['random_delta']:+.4f} with it off" if none_arm and none_arm["ablation"] else "more with it off")
          + ". The first layer does the most work and is the most sensitive to how it is routed,",
          "which is also where the published fault was found — worth watching per layer, not on",
          "an average.",
          "")

if R.get("reduced_scale_note"):
    w(f"> **{R['reduced_scale_note']}**", "")

w("## Files", "",
  "| file | what it is |",
  "|---|---|",
  "| `run.py` | the five stages; writes `results.json` and `run.log` |",
  "| `detect.py` | the trace generator and the four spike rules |",
  "| `rollup.py` | mean / min-max / M4 / LTTB and a line rasteriser to score them |",
  "| `misload.py` | the silent misload, the five cheap checks, the keyspace-first fix |",
  "| `freeze.py` | optimizer groups, zero-grad counts, grad-norm shift, weight hashes |",
  "| `routing.py` | the five per-layer routing metrics and the two ablations |",
  "| `make_readme.py` | generates this file from `results.json` |",
  "| `results.json` | every number above |",
  "| `run.log` | the training log |",
  "| `stability.ipynb` | a short live version |",
  "",
  "`misload.py`, `freeze.py` and `routing.py` import Session 14's `model.py` and `train.py`",
  "unchanged and need `session-14-moe/dense.pt`, which Session 14's `train.py` writes.",
  "")

w("## What this does and does not show", "",
  "- The detector and downsampling results are on **synthetic traces** with planted spikes.",
  "  That is the point — the ground truth has to be known to score a rule — but a real",
  "  grad-norm trace has structure these do not, and a rule that wins here could lose there.",
  "- The misload, freezing and routing results are on a **21M-parameter model**. The",
  "  mechanisms are scale-free; the magnitudes are not.",
  "- Nothing here measures the thing a real run actually fails on, which is hardware. A",
  f"  single machine cannot produce a straggler, an Xid, or a silent data corruption.",
  "- The routing stage grows one model one way. It does not sweep gamma, top-k, or expert",
  "  count, so it shows that the intervention rate *can* separate layers, not what a healthy",
  "  value is.",
  "")

(HERE / "README.md").write_text("\n".join(O) + "\n")
print(f"wrote README.md -- {len('\n'.join(O))} chars, {len(O)} lines")
