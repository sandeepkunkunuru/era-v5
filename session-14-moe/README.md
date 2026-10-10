# Session 14 — train a dense model, grow it into a mixture of experts

> **Assignment.** Train a Linear model and convert that into an MoE! Your call on model size and
> data trained on, but must show they continue to train and reduce loss!
>
> The assignment page adds one requirement: *the repo MUST have training logs.* They are in
> [`run.log`](run.log), and every number below is read from [`results.json`](results.json) by
> `make_readme.py`.

A dense byte-level transformer is trained, then converted into a mixture-of-experts model **four
different ways**, and each converted model is trained further against a dense control that gets the
same tokens. The conversion is built so that the mixture-of-experts model uses **exactly the same
width per token** as the dense model, which makes the comparison about where the parameters sit and
nothing else.

```
dense   21,538,176 parameters   one feed-forward network per layer, inner width 1024
MoE     64,152,960 total        32 experts of inner width 128, top-8
        21,685,632 active       8 x 128 = 1024 = the dense inner width
```

| | dense | MoE | ratio |
|---|--:|--:|--:|
| total parameters | 21,538,176 | 64,152,960 | 2.98x |
| active parameters per token | 21,538,176 | 21,685,632 | 1.01x |

That is the session's whole claim in two rows: 3.0 times the stored
parameters for the same work per token.

## How to run it

```bash
python data.py              # 50M train / 2M val bytes from Session 4's cleaned shard
python train.py --quick     # a few minutes; proves the plumbing
python train.py             # the real run; writes run.log and results.json
python make_readme.py       # regenerates this file
python make_notebook.py     # builds moe.ipynb and executes it
```

Run on **NVIDIA GeForce RTX 4050 Laptop GPU**, torch 2.12.1+cu130, seed 20261010, block 512,
batch 8 x 2 accumulation steps.

## The shape, and why these numbers

The backbone is Session 13's: 12 layers, d_model 384, 6 heads,
block 512, byte vocabulary (256), SwiGLU feed-forward. The inner width works out to
1024, which cuts into exactly 8 slices of
128 — and that is what makes growth arithmetic instead of guesswork. With
32 experts the layer stores 4
tilings of the dense network's neurons.

The four growth methods, following Session 14's S15 and the V4 cookbook:

| method | how the experts start |
|---|---|
| `random` | fresh weights; the control, nothing is inherited |
| `clone` | 8 distinct slices, each copied 4 times -> 8 clone families |
| `partition` | the 1024 neurons permuted per tiling and cut into experts of 128 |
| `drop` | `partition` with half of each expert's neurons redrawn (drop-upcycling, r = 0.5) |

True sparse upcycling — every expert a *full* copy of the feed-forward network — needs the expert
width to equal the dense width, which this shape rules out by construction. `clone` is the
within-shape version of it, and it is also the case the V4 cookbook reports as a failure: a router
cannot tell near-identical clones apart.

## 1. The dense model

| tokens | steps | val loss | final train loss | tok/s | peak MiB |
|--:|--:|--:|--:|--:|--:|
| 11,993,088 | 1,464 | **1.1881** | 1.165 | 63,240 | 1,355 |

This is the model every conversion starts from. 11,993,088 tokens of bytes is not
a trained language model, but it is well past the point where the loss is dominated by the unigram
byte distribution, so a conversion that damages it shows up immediately.

## 2. The conversion can be exact

Force the router onto the inherited experts (zero its weights and give the first top-k experts a
choosing bias) and scale the inherited down-projection by top-k. Then the mixture-of-experts layer
should *be* the dense layer. Measured against the dense model's validation loss:

| method | val loss | delta |
|---|--:|--:|
| `clone` | 1.188146 | +0.000046 |
| `partition` | 1.188146 | +0.000046 |
| `drop` | 3.074660 | +1.886560 |

`clone` and `partition` reproduce the dense model to float precision: their experts together hold
exactly the dense network's neurons, and a weighted sum with weights adding to one, times top-k,
puts the layer back together. `drop` cannot, because half of each expert is new — the delta is the
price of the redrawn neurons, paid once.

This matters because it separates two things that are easy to confuse: damage from the *structure*
of the conversion, and damage from the *initialisation* of the router.

## 3. The loss at the moment of conversion, and the scaling factor nobody explains

With a real (freshly initialised) router instead of a forced one, the top-k weights are all about
1/k, so the layer's output is about 1/k of the dense layer's unless the down-projection is scaled
back up. The brief mentions a *routed scaling factor* — 2.5 in DeepSeek-V3, 1.5 in V4.1-Flash — and
says DeepSeek gives the value without an explanation. In this shape the right value is known
exactly: it is top-k.

| method | scale_down | val loss | delta against dense |
|---|---|--:|--:|
| `random` | none | 4.2154 | +3.0273 |
| `random` | x8 | 4.1863 | +2.9982 |
| `clone` | none | 4.3126 | +3.1245 |
| `clone` | x8 | 1.845 | +0.6569 |
| `partition` | none | 4.2397 | +3.0516 |
| `partition` | x8 | 1.9509 | +0.7628 |
| `drop` | none | 4.2007 | +3.0126 |
| `drop` | x8 | 3.3322 | +2.1441 |

Without the scaling, every inherited conversion costs about
+3.06 nats on the spot. With it, `clone` and
`partition` cost about +0.710. That is a factor of
4
difference from one multiplication, and it is the difference between a conversion that continues
from where the dense model stopped and one that throws away a chunk of the training that paid for
it.

Each method carries forward the scaling that measured better:
`random` x8, `clone` x8, `partition` x8, `drop` x8. For `random`
the setting is a no-op — there is nothing inherited to scale — so its two rows differ only by the
initialisation draw, and the gap between them is a reading of the measurement's own noise.

## 4. The assignment's actual question: does it keep training, and does the loss drop?

Yes, for every inherited init. Each model gets 8,000,000 further tokens, and the
dense control gets the same.

| run | init | val loss after | val loss before | change | MaxVio | dead | tok/s |
|---|---|--:|--:|--:|--:|--:|--:|
| `dense-control` | dense, unchanged | **0.9768** | 1.1881 | -0.2113 | -- | --/-- | 62,580 |
| `grow:random` | `random` | **1.0377** | 4.1863 | -3.1486 | 0.0154 | 0/384 | 14,424 |
| `grow:clone` | `clone` | **0.9845** | 1.8450 | -0.8605 | 0.0139 | 0/384 | 14,432 |
| `grow:partition` | `partition` | **0.991** | 1.9509 | -0.9599 | 0.0138 | 0/384 | 14,582 |
| `grow:drop` | `drop` | **0.989** | 3.3322 | -2.3432 | 0.0149 | 0/384 | 14,569 |

**What the numbers say.**

- Every converted model trains. Loss falls monotonically over the continuation for all four inits;
  the per-run curves are in `results.json` under `curve`.
- **Upcycling is worth real compute.** `random` — the same architecture with nothing inherited —
  ends at 1.0377 against
  0.991 for `partition`, on identical tokens. The inherited
  weights are not a convenience; they are most of the result at this budget.
- **The dense control is the honest comparison and it is close.** The control ends at
  0.9768 against 0.9845 for the best
  conversion (`clone`). At 8,000,000 tokens the mixture-of-experts model has
  3.0x the parameters and has not yet had the tokens to use them —
  which is the session's own point about capacity following the total and needing data to fill it.
- **Throughput is the cost.** 62,580 tok/s dense against
  about 14,432 tok/s for the mixture of experts, at equal
  active parameters. The gap is the dispatch loop: 32 small matrix multiplications per layer instead
  of one large one, on one GPU, with no expert parallelism and no grouped kernel. This is exactly
  the cost the brief's Section 16 is about, and on one card there is nowhere to hide it.

## 5. Balancing: what stops experts dying

Same init (`partition`), same tokens, three balancing methods. MaxVio is
`(max load - mean) / mean` over the last fifth of training, averaged over layers; a dead expert is
one that received no tokens in that window.

| balancing | val loss | MaxVio (mean) | MaxVio (worst layer) | dead experts |
|---|--:|--:|--:|--:|
| none | 0.9836 | 2.1498 | 3.0 | 61/384 |
| Switch auxiliary loss, alpha = 0.01 | 0.9844 | 1.8435 | 3.0 | 0/384 |
| auxiliary-loss-free bias, gamma = 0.001 | 0.991 | 0.0138 | 0.0235 | 0/384 |

**The ordering is the one the session teaches, and the loss column is why the session says to look
past it.** Unbalanced training reaches a validation loss of
0.9836 — no worse than the balanced runs at this budget — while
leaving 61 experts with no tokens at all and a MaxVio of
2.1498. The bias method holds MaxVio at
0.0138 with 0 dead.

That is the lecture's *"dropping the loss immediately is not what we want, we want the eventual loss
to be low"* in one table: a dead expert costs nothing today and costs its whole capacity for the
rest of training. Switch's auxiliary loss sits between the two, and its gradient goes into the same
router weights the language loss is training, which is the reason DeepSeek-V3 and V5 drop it.

## 6. Clone families, and the fix

The V4 cookbook reports a failure the published growth studies do not: when experts are cloned
without redrawing any neurons, *the router cannot tell the clones apart*, and each family collapses
onto a few members — 27 dead experts rising to 168 of 460 within a few hundred steps. Its fix was to
replace hard top-k with probabilistic selection during the early window, so that every clone
receives some gradient before the router settles.

Reproduced here on the `clone` init, with **balancing switched off** in both arms, because with the
bias on it is the bias that keeps the clones alive and the routing rule cannot be seen:

| routing | val loss | MaxVio | dead experts |
|---|--:|--:|--:|
| hard top-k | 0.9773 | 2.2058 | 77/384 |
| probabilistic for 200 steps | 0.9889 | 1.9728 | 14/384 |

Probabilistic selection samples k experts with probability proportional to their score instead of
taking the k highest, so a clone with a slightly lower score still gets tokens sometimes. The point
is not the loss — it is the dead-expert column, which is capacity that never comes back.

## 7. The balancing window

The bias update needs a load count, and the count covers some window. V4 counted each micro-batch;
Qwen's 2025 measurement, in the brief's Section 14, says counting the whole batch is better because
a micro-batch of one kind of text should not be forced to use every expert evenly.

| window | val loss | MaxVio | dead experts |
|---|--:|--:|--:|
| the whole batch | 0.991 | 0.0138 | 0/384 |
| each micro-batch | 0.9896 | 0.0124 | 0/384 |

At 2 accumulation steps the two windows differ by a factor of 2, which is as
small a difference as this experiment can make; Qwen's comparison was 4 or 16 sequences against 512.
The measurement is here for completeness and the difference is within run-to-run noise at this
scale — the honest reading is that this question needs the micro-batches to be specialised by
content, which one shuffled byte stream cannot provide.

## What this does and does not show

**Shows:** a dense model can be converted into a mixture of experts without losing what it learned,
provided the output is scaled for the router's weights; the converted model keeps training and the
loss keeps falling; inheriting weights beats starting the experts from scratch by a wide margin at
this budget; and balancing is what keeps experts alive, measured by MaxVio and dead counts rather
than asserted.

**Does not show:** anything about expert parallelism. One GPU holds every expert, so there is no
dispatch all-to-all, no load imbalance across devices, and no communication cost — the three things
the brief's Sections 16 and 17 are about. The throughput gap in Section 4 is a single-GPU dispatch
artifact, not the all-to-all cost a real run pays. It also does not show subject-level expert
specialisation: that needs a mixed corpus and far more tokens than this.

**One honest caveat about the loss comparisons.** The dense control and the converted models each
restart their learning-rate schedule at the conversion point, with the same peak and the same warm-up,
so they are comparable to each other but neither continues the dense model's original schedule.
Session 11's learning-rate rule was not re-applied for the changed effective batch, because the batch
does not change here.

## Files

| file | what |
|---|---|
| `data.py` | builds `train.bin` / `val.bin` from Session 4's cleaned parquet shard |
| `model.py` | the dense GPT, the MoE layer (router, dispatch, balancing), and `upcycle()` |
| `train.py` | the seven stages; writes `run.log` and `results.json` |
| `make_readme.py` | generates this file from `results.json` |
| `make_notebook.py` | builds and executes `moe.ipynb` |
| `run.log` | **the training log the assignment requires** |
| `results.json` | every measurement, including the per-run loss curves |

The session's own material (the written brief, the lecture transcript and our notes on where the
two disagree) is third-party course content and stays in a private repo; this directory is the
code and the measurements.

