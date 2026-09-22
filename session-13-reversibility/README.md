# ERA V5 · Session 13 — reversibility, measured

A 21.2M-parameter byte-level GPT trained on 50M tokens, four times: once normally, once with a reversible stack at the same batch size, once with the reversible stack at the largest batch it allows, and once more at that batch with the learning rate corrected for it. Everything below came out of one run of [`train.py`](train.py); [`run.log`](run.log) is its transcript and [`results.json`](results.json) holds the numbers. This file is generated from that JSON.

```bash
python data.py      # build the corpus from Session 4's cleaned shard
python train.py     # runs 1-3, plus the checks below
python extra_run.py # run 4: the same batch as run 3 with the learning rate scaled
python train.py --quick
```

Setup: NVIDIA GeForce RTX 4050 Laptop GPU, torch 2.12.1+cu130, seed 20260919, sequence length 512, byte vocabulary (256), AdamW at 3e-4, gradient clipping 1.0, **dropout 0 and weight decay 0 in every arm** — see "Why dropout must be zero" below.

## The runs

| | stack | batch | optimizer steps | tokens/s | peak memory | val loss |
|---|---|--:|--:|--:|--:|--:|
| baseline | baseline | 26 | 3,756 | 22,429 | 5,243 MiB | 0.7984 |
| reversible-same-batch | leapfrog | 26 | 3,756 | 13,977 | 1,069 MiB | 0.8438 |
| reversible-max-batch | leapfrog | 153 | 638 | 13,997 | 4,565 MiB | 1.3244 |
| reversible-max-batch-lr-scaled | leapfrog | 153 | 638 | 13,916 | 4,565 MiB | 1.0306 |

**Reading it.** At the same batch size the reversible stack is **37.7% slower** than the baseline and uses **79.6% less memory** — which is the trade the method exists to make, and the paper prices the slowdown at 30–50%. Spending that memory on a bigger batch (153 against 26) leaves it **37.6% slower** than the baseline. Validation loss: 0.7984 against 0.8438 at equal batch — +0.0454 nats, one seed, no other difference.

**The bigger batch bought no speed.** 13,997 tokens/s at batch 153 against 13,977 at batch 26 — a 0.1% difference, which is nothing. The paper reports throughput gains up to 101% from exactly this move, but its gains come at 96 layers, where the activation term dominates. At 12 layers on a 6 GB card there is nothing left to win: the run is bound by the recomputation, not by how many sequences are in flight.

**And most of run 3's loss penalty was the learning rate, not the batch.** Run 3 changes two things at once: the batch rises 5.9x, so the same 50M tokens are covered in 638 optimizer steps instead of 3,756 — at a learning rate chosen for the small batch. Run 4 repeats run 3 with the learning rate scaled by the square root of the batch ratio (3.0e-04 → 7.28e-04, the rule from Session 11) and nothing else changed: validation loss goes 1.3244 → 1.0306, recovering 0.29 of the 0.48 nat gap. A fixed token budget spent in fewer, larger steps still costs something — but most of what looked like the cost was an untuned comparison.

**A note on that batch size.** The probe found 180 survivable — three full steps, clipping and optimizer included — and training at 180 still ran out of memory, so the run backed off to **153**. A short probe cannot see the allocator's steady state on a card this full; the number in the table is the one that actually trained 50M tokens.

## Which variant works

The class names two variants, "Euler and midpoint". The paper (arXiv:2512.02056) gives three, and plain forward Euler is not one of them — it is not reversible, because recovering the previous state would need the block evaluated at the state you are trying to recover. What it actually offers is:

| variant | rule | inverse |
|---|---|---|
| midpoint (eq. 2.4) | `p[l+1] = a·p[l-1] + 2h·f(p[l])` | `p[l-1] = (p[l+1] - 2h·f(p[l])) / a` |
| leapfrog (eq. 2.6) | `p[l+1] = 2p[l] - p[l-1] + h²·f(p[l])` | `p[l-1] = 2p[l] - p[l+1] + h²·f(p[l])` |
| Hamiltonian (eq. 2.8-2.9) | `q[l] = a·q[l-1] + Attn(LN(p[l-1]))`, `p[l] = a·p[l-1] + MLP(LN(q[l]))` | undo the MLP step, then the attention step |

Short runs at equal batch, 4.0M tokens each:

| variant | tokens/s | peak memory | val loss |
|---|--:|--:|--:|
| midpoint | 13,822 | 1,069 MiB | 2.4429 |
| leapfrog | 13,622 | 1,069 MiB | 2.4033 |
| hamiltonian | 16,790 | 932 MiB | 2.4651 |

**leapfrog wins on validation loss**, and is what the three headline runs use.

## Is it actually reversible?

Two checks, because a reversible stack that does not reconstruct exactly is just a slower model with wrong gradients.

**The custom backward against ordinary autograd**, in fp64, worst relative error over every parameter:

| variant | worst relative gradient error |
|---|--:|
| midpoint | 8.22e-16 |
| leapfrog | 6.61e-16 |
| hamiltonian | 1.21e-15 |

**Reconstruction error**, and what V4's damping coefficient does to it. The paper's stability analysis (§3) requires |a| = 1 for a method to be stable forwards *and* backwards; V4's production integrator shipped **a = 0.5**:

| variant | a = 1.0 (paper) | a = 0.5 (V4's setting) |
|---|--:|--:|
| midpoint | 2.99e-14 | 2.97e-11 |
| leapfrog | 6.98e-15 | 6.98e-15 |
| hamiltonian | 1.08e-13 | 1.36e-06 |

At a = 1 every stack rebuilds to floating-point noise. At a = 0.5 the midpoint inverse divides by 0.5 at every layer, so any error doubles on the way down and the relative error is **1e+03× larger**; the Hamiltonian stack, which divides by a twice per layer, is worse still. **Leapfrog is unaffected, because its rule has no a in it.** V4 shipped a = 0.5, and the paper's §3 requires |a| = 1 for a method to be stable in both directions — this table is what that requirement costs when it is ignored.

## Memory against depth

The paper's headline structural claim is that a reversible stack's activation memory does not grow with depth, while a standard stack's does.

| layers | parameters | baseline peak | reversible peak |
|--:|--:|--:|--:|
| 4 | 7.5M | 922 MiB | 438 MiB |
| 8 | 14.6M | 1,756 MiB | 492 MiB |
| 12 | 21.6M | 2,590 MiB | 546 MiB |
| 24 | 42.9M | 5,091 MiB | 841 MiB |
| 48 | 85.4M | **out of memory** | 1,651 MiB |

Measured at a fixed batch: the baseline costs about **208 MiB per layer** and the reversible stack about **20 MiB per layer** — the second number is weights and optimizer state, which both stacks pay, and nothing else. At 48 layers the baseline cannot run on this card at all while the reversible stack uses 1,651 MiB. That is the paper's claim, and it is the reason the method exists: the saving is in the term that scales with depth.

## Why dropout must be zero

The backward pass rebuilds each layer's input by running the block again. With dropout on, the second run draws a different random mask, so it reconstructs a state the forward pass never produced and the gradients are wrong. This is a **correctness** requirement, not a preference — and it means the baseline has to run at dropout 0 too, or the comparison is between two different models rather than two stacks.

The class also said weight decay cannot be used with reversibility [01:29:57]. The paper does not say that, and Session 12's brief recorded V4's zero weight decay as a deliberate architectural choice. These runs use weight decay 0 in every arm so the arms stay comparable, but the claim itself is still unsourced.

## What this cost

All runs on one NVIDIA GeForce RTX 4050 Laptop GPU: 3.8 GPU-hours in total. On a rented A100 at about $1.50/hour that would be roughly $5.75 of compute; the point of the exercise is that the reversible arm buys its extra time back only when the recovered memory is spent on a bigger batch.

## Honest limits

- One seed, one model size, one sequence length. The direction is measured; the magnitudes are not a scaling law.
- 21.2M parameters is small enough that weights and optimizer state dominate peak memory, which compresses the difference this method exists to create. The effect grows with depth and sequence length, and this GPU has 6 GB.
- Throughput here is wall-clock on a laptop GPU with other processes idle; it is not a clean benchmark.
- The first full run finished every training run and then crashed in the depth sweep before writing its JSON. The numbers for 6 runs were parsed back out of `run.log` rather than re-running three GPU-hours; their per-step loss curves are gone and `final_train_loss` is null in `results.json`. Validation loss, throughput and peak memory are the logged values. `train.py` now saves after every stage and treats an out-of-memory error in the sweep as a result.
- The corpus is the Session 4 cleaned shard, read as raw bytes. A byte vocabulary keeps the embedding out of the parameter budget, but it also makes the loss numbers incomparable with any BPE-tokenized run (Session 9's point about perplexity across tokenizers).
