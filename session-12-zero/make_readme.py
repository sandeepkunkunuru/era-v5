"""Generate README.md from results.json, so the write-up cannot drift from the run."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
N, P = R["N"], R["P_bytes"]
W = max(R["worlds"])
runs = {(s["W"], s["stage"]): s for s in R["runs"]}
top = [runs[(W, st)] for st in range(4)]
theory = lambda st, w: {0: 16.0, 1: 4 + 12 / w, 2: 2 + 14 / w, 3: 16 / w}[st]
GIB = 2 ** 30
out = []
w = out.append

w(f"# ERA V5 · Session 12 — ZeRO on {W} virtual GPUs, written out by hand\n")
w(f"{W} CPU processes stand in for {W} GPUs. They talk through `torch.distributed` (the gloo backend) using the "
  "same three collectives a real run uses — all-reduce, reduce-scatter and all-gather. A tiny GPT trains on top of "
  "them under data parallelism and under ZeRO stages 1, 2 and 3.\n")
w("There is no DeepSpeed and no FSDP here. Each stage is a few dozen lines that call the collectives in the right "
  "order, so the mechanism is readable in [`zero_sim.py`](zero_sim.py) rather than hidden in a library. Every number "
  "on this page came out of one run: [`run.log`](run.log) is its transcript, [`results.json`](results.json) holds the "
  "numbers, and this file is generated from that JSON by `make_readme.py`.\n")
w("```bash\npython zero_sim.py            # stages 0-3 at 8, 16 and 32 virtual GPUs, then the single-process check\n"
  "python zero_sim.py --quick    # 8 virtual GPUs, fewer steps\n"
  "python make_readme.py         # regenerate this file\n```\n")
w(f"The notebook [`zero_sim.ipynb`](zero_sim.ipynb) runs the same code cell by cell. Setup: torch {R['torch']}, "
  f"{R['cpu_threads']} CPU threads, seed {R['seed']}, {R['steps']} steps per run.\n")

# ---- at a glance
w(f"## The answer, at {W} GPUs\n")
w("| | stage 0 (data parallel) | stage 1 | stage 2 | stage 3 |\n|---|--:|--:|--:|--:|")
w("| what is split across GPUs | nothing | optimizer state | + gradients | + weights |")
w("| memory per GPU, bytes per parameter (measured) | " + " | ".join(f"**{s['bytes_per_param']:.2f}**" for s in top) + " |")
w("| the same, from the formula | " + " | ".join(f"{theory(s['stage'], W):.2f}" for s in top) + " |")
w("| traffic per GPU per step, in P | " + " | ".join(f"{s['traffic_total_P']:.3f}" for s in top) + " |")
w("| optimizer step, ms per GPU | " + " | ".join(f"{1000 * s['time']['optimizer']:.2f}" for s in top) + " |")
w("| loss difference from one process on the whole batch | "
  + " | ".join(f"{R['loss_diff_vs_reference'][str(s['stage'])]:.1e}" for s in top) + " |")
w("")
w(f"P is the size of the whole model in bf16: {N:,} parameters × 2 bytes = {P:,} bytes. In one sentence: **each "
  f"stage stops keeping one more thing on every GPU, memory falls from 16 bytes a parameter to {top[3]['bytes_per_param']:.2f}, "
  "stages 1 and 2 send exactly what data parallelism already sends, stage 3 sends half as much again — and the model "
  "learns exactly the same thing in all four.**\n")

# ---- 16 bytes
w("## Why 16 bytes a parameter\n")
w("Training in bf16 with Adam keeps five numbers for every weight:\n")
w("| what | format | bytes | why it is there |\n|---|---|--:|---|")
w("| the weight used for arithmetic | bf16 | 2 | the forward and backward passes run on it |")
w("| its gradient | bf16 | 2 | backward writes it; the optimizer reads it |")
w("| a master copy of the weight | fp32 | 4 | adding a tiny update to a bf16 number rounds it away; the fp32 copy keeps it |")
w("| Adam's running mean of the gradient, m | fp32 | 4 | the direction |")
w("| Adam's running mean of the squared gradient, v | fp32 | 4 | the per-weight scale |")
w("| **total** | | **16** | |")
w("")
w("Under plain data parallelism every GPU holds all 16 bytes for every weight, and all the copies are identical. ZeRO "
  "(\"zero redundancy\") is the observation that most of that duplication is never needed.\n")

# ---- the stages
w("## The four stages, as the collectives each GPU calls\n")
w("The three collectives, for W GPUs each holding a tensor:\n")
w("- **all-reduce** — everyone ends with the sum of everyone's tensor. A ring implementation sends 2(W−1)/W of the "
  "tensor per GPU.\n- **reduce-scatter** — everyone ends with *one 1/W slice* of the sum. (W−1)/W per GPU.\n"
  "- **all-gather** — everyone starts with one slice and ends with all of them. (W−1)/W per GPU.\n")
w("A reduce-scatter followed by an all-gather *is* an all-reduce, done in two halves. That equivalence is the whole "
  "reason stages 1 and 2 cost nothing extra.\n")
w("Every parameter is flattened, padded to a multiple of W, and cut into W equal slices; GPU *r* owns slice *r* of "
  "every parameter (`padded()` and `chunk()` in the code).\n")

w("### Stage 0 — data parallelism\n")
w("Each GPU holds everything, reads a different slice of the batch, and computes its own gradients. Then, per "
  "parameter: `all_reduce(grad)` and divide by W. Every GPU now holds the same averaged gradient, runs the same Adam "
  "update on its full fp32 copy, and ends with the same weights. The duplication is total: W copies of 16 bytes a "
  "weight.\n")
w("### Stage 1 — split the optimizer state\n")
w("The fp32 master copy, m and v are 12 of the 16 bytes, and they are only ever used by the update. So each GPU keeps "
  "them for its own slice only. After backward, instead of an all-reduce: `reduce_scatter(grad)` — each GPU receives the "
  "averaged gradient for *its* slice, which is all its optimizer needs. It updates that slice, casts it to bf16, and "
  "`all_gather` puts the full bf16 weights back on every GPU. Those two collectives are the two halves of the "
  "all-reduce stage 0 was already doing, so the traffic is identical.\n")
w("### Stage 2 — split the gradients too\n")
w("A GPU only needs the gradient for its own slice, so there is no reason to keep the rest. The code registers a hook "
  "on every parameter (`register_post_accumulate_grad_hook`) that fires the moment backward has finished that "
  "parameter's gradient: it reduce-scatters it straight away and drops the full-size gradient. The full set of "
  "gradients therefore never exists at once — the most that exists is one parameter's, which is the transient below. "
  "Same collectives, same traffic, two more bytes saved.\n")
w("### Stage 3 — split the weights too\n")
w("Now each GPU holds only its slice of the bf16 weights as well. A layer's full weights are gathered just before the "
  "layer runs and thrown away just after. This is the only stage where the model code had to change, and the reason is "
  "worth stating: an ordinary `x @ W.T` makes autograd *save W* for the backward pass, which would keep every layer's "
  "full weights alive until backward and save nothing. So the linear layer is a custom autograd function (`ZLinear`) "
  "that saves only the slice:\n")
w("- **forward**: all-gather W → compute y = x Wᵀ → free W\n"
  "- **backward**: all-gather W again → compute dL/dx and dL/dW → reduce-scatter dL/dW into slices → free both\n")
w("That second gather is the extra traffic: stage 3 pays a gather in forward, a gather in backward and the "
  "reduce-scatter, which is 3 × (W−1)/W of P rather than 2. The measurement comes in just under that, and the reason "
  "is visible in the call counts: an embedding's backward needs only the token ids, not the table, so the two "
  "embeddings are gathered once instead of twice (`ZEmbed`). There is also no all-gather after the optimizer step — "
  "the slice a GPU just updated *is* the weight it stores.\n")

# ---- memory
w("## Memory, measured\n")
w("After a step, each GPU walks the tensors it actually holds and adds up their bytes by category. Nothing below is "
  "computed from a formula; the formula is the second column, for comparison.\n")
w("![memory per GPU by stage](figures/memory_by_stage.png)\n")
w("| GPUs | stage | weights | gradients | fp32 copy + m + v | bytes / parameter | formula | largest transient |\n"
  "|--:|--:|--:|--:|--:|--:|--:|--:|")
for s in R["runs"]:
    m = s["mem"]
    w(f"| {s['W']} | {s['stage']} | {m['weights_bf16']:,} | {m['grads_bf16']:,} | "
      f"{m['master_fp32'] + m['adam_m'] + m['adam_v']:,} | {s['bytes_per_param']:.2f} | {theory(s['stage'], s['W']):.2f} | "
      f"{s['transient_peak']:,} |")
w("")
w("(bytes per GPU; the fullest GPU is reported, which only matters for the padding of the last slice.)\n")
w("![memory against the number of GPUs](figures/memory_vs_world.png)\n")
w("Stage 0 never moves off 16 however many GPUs there are. Stages 1 and 2 fall towards a floor — 4 and 2 bytes, the "
  "parts that are still copied everywhere. Only stage 3 keeps falling as 16/W.\n")
w(f"**The transient column is the part the formula leaves out.** Stage 2's is one full gradient — the largest "
  f"parameter's — in the moment between backward writing it and the hook sending it. Stage 3's is one gathered layer "
  f"plus its full gradient, in backward. At {W} GPUs this tiny model's stage-3 transient "
  f"({top[3]['transient_peak']:,} bytes) is bigger than everything the GPU keeps permanently "
  f"({top[3]['persistent']:,} bytes). ZeRO-3's floor is one layer, not zero. In a large model one layer is a small "
  "slice of the whole and this stops mattering; activations, which ZeRO does not touch at all, matter much more.\n")

# ---- traffic
w("## Communication, measured\n")
w("Every collective goes through a counter (`Ledger`) that records its payload using the ring volumes above; one "
  "full step is counted, after a warm-up step.\n")
w("| GPUs | stage | all-reduce | reduce-scatter | all-gather | total, in P | formula | calls |\n|--:|--:|--:|--:|--:|--:|--:|---|")
for s in R["runs"]:
    t, c = s["traffic_P"], s["calls"]
    f = (3 if s["stage"] == 3 else 2) * (s["W"] - 1) / s["W"]
    calls = ", ".join(f"{k.replace('_', '-')} {v}" for k, v in c.items() if v)
    w(f"| {s['W']} | {s['stage']} | {t['all_reduce']:.3f} | {t['reduce_scatter']:.3f} | {t['all_gather']:.3f} | "
      f"**{s['traffic_total_P']:.3f}** | {f:.3f} | {calls} |")
w("")
w("The brief rounds these to 2P and 3P; the exact ring figures are 2(W−1)/W and 3(W−1)/W, which is why they creep up "
  "towards 2 and 3 as W grows. Stage 3 sits just under its formula because the two embeddings skip the backward "
  "gather — 9 backward gathers for 11 parameters.\n")

# ---- time
w("## Computation, measured — and what it does and does not show\n")
w("![time per step by stage](figures/time_by_stage.png)\n")
w("| GPUs | stage | forward + backward | inside collectives | optimizer | total, ms |\n|--:|--:|--:|--:|--:|--:|")
for s in R["runs"]:
    t = s["time"]
    w(f"| {s['W']} | {s['stage']} | {1000 * t['compute']:.1f} | {1000 * t['comm']:.1f} | {1000 * t['optimizer']:.2f} | "
      f"{1000 * t['total']:.1f} |")
w("")
s0, s1 = top[0]["time"]["optimizer"], top[1]["time"]["optimizer"]
w(f"**What changes, and why.** The forward and backward arithmetic is the same in every stage — ZeRO does not change "
  f"how many multiplications the model needs. What changes is the optimizer: from stage 1 on, each GPU updates 1/W of "
  f"the parameters, and at {W} GPUs its step falls from {1000 * s0:.1f} ms to {1000 * s1:.2f} ms. That is "
  f"{s0 / s1:.0f}× rather than {W}×, because a fixed per-call cost does not shrink with the slice. Stage 3 also moves "
  "communication *into* the forward and backward passes: a layer cannot run until its weights have arrived, which is "
  "why real systems prefetch the next layer's gather while the current one computes.\n")
w("**What this simulation cannot show.** The time bars are not a prediction for GPUs. Here 32 processes share "
  f"{R['cpu_threads']} CPU threads, the collectives cross local sockets, and every parameter is sent in its own small "
  "collective, so the cost is dominated by per-call latency rather than bandwidth. That is why stages 1 and 2, which "
  "make two calls per parameter where stage 0 makes one, come out *slower* here although they move the same bytes. A "
  "real system groups gradients into buckets of a few hundred megabytes and overlaps them with backward, which is "
  "exactly the fix for this. The memory and traffic numbers do transfer to GPUs; the milliseconds do not.\n")

# ---- correctness
ref = R["reference_losses"]
w("## The sharding did not change what the model learned\n")
w(f"The same model was trained once more in a single process on the whole global batch ({W} × {R['config']['B']} "
  f"sequences). If data parallelism is \"mathematically identical to one GPU on a batch W times larger\", and if ZeRO "
  "only rearranges storage, all five loss curves must coincide.\n")
w("| step | one process | " + " | ".join(f"stage {s['stage']}" for s in top) + " |\n|--:|--:|" + "--:|" * 4)
for i in range(len(ref)):
    if i in (0, 1, 3, 7) or i == len(ref) - 1:
        w(f"| {i + 1} | {ref[i]:.4f} | " + " | ".join(f"{s['losses'][i]:.4f}" for s in top) + " |")
w("")
w("The largest difference over the run is "
  + ", ".join(f"{R['loss_diff_vs_reference'][str(s['stage'])]:.1e} (stage {s['stage']})" for s in top)
  + ". What is left is bf16 rounding: summing the same gradients in a different order rounds differently.\n")

# ---- 30B
w("## Applying the measured rule to V5's 30B model\n")
w("The measured bytes per parameter follow 16, 4 + 12/W, 2 + 14/W and 16/W exactly, so they can be carried to a "
  "30-billion-parameter model (training state only, no activations):\n")
w("| GPUs | stage 0 | stage 1 | stage 2 | stage 3 |\n|--:|--:|--:|--:|--:|")
for g in (8, 16, 32, 64):
    cells = []
    for st in range(4):
        gib = 30e9 * theory(st, g) / GIB
        cells.append(f"{gib:.1f} GiB" + (" ✓" if gib <= 74.5 else ""))
    w(f"| {g} | " + " | ".join(cells) + " |")
w("")
w("✓ = fits an 80 GB card (74.5 GiB). This reproduces the brief's memory ladder exactly. Stages 0 and 1 never fit, "
  "because the weights and gradients alone are 4 bytes a parameter — 111.8 GiB — on every card however many there "
  "are. Stage 2 fits from 32 GPUs, stage 3 from 8, and in both cases the room left over has to hold the activations.\n")

# ---- understanding
w("## What I take away\n")
w("- **Data parallelism wastes memory on purpose, to keep the maths trivial.** Every GPU is a full copy, so the only "
  "coordination needed is one all-reduce of the gradients.\n"
  "- **ZeRO is the same arithmetic with the duplicates deleted.** Stages 1 and 2 keep only what a GPU updates, and pay "
  "for it with the reduce-scatter and all-gather that the all-reduce was already made of — so for 10 of the 16 bytes "
  "they cost nothing in traffic.\n"
  "- **Stage 3 is a different trade.** It removes the last duplicate, the weights, at the price of 50% more traffic "
  "and of putting communication on the critical path of every layer. It only makes sense when the model does not fit "
  "otherwise — which at 30B on 8 GPUs it does not.\n"
  "- **None of it changes the model.** The loss curves agree with a single process to bf16 rounding.\n"
  "- **The formula is not the whole bill.** Stage 3 needs room for one gathered layer, and every stage needs room for "
  "activations, which ZeRO does not shard at all.\n")
w("## Limits of this simulation\n")
w("- The GPUs are CPU processes and the model is 0.4M parameters, so the timings show structure, not speed.\n"
  "- No bucketing and no overlap: each parameter is its own collective, issued after it is ready. Real ZeRO buckets "
  "and overlaps both.\n"
  "- Activation memory is not measured, because ZeRO leaves it unchanged across stages.\n"
  "- Gradients are reduced in bf16, so P and the reduced tensors are the same size; systems that reduce in fp32 send "
  "twice as much.\n")
(HERE / "README.md").write_text("\n".join(out))
print(f"README.md: {len(chr(10).join(out).split())} words")
