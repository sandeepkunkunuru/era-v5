# Session 15 — a training run that watches itself

> Session 15's assignment is a team-selection form, not a build. So this answers the
> session's own open questions instead: of the four spike rules it lists, which one
> would you actually page a human with? Does M4 downsampling really lose nothing? How
> small can a silent checkpoint misload get before no number can see it? And what does
> the top-k intervention rate show that MaxVio and dead-expert counts cannot?

Five stages, every number below read out of `results.json`, which is written by
`run.py` and logged to `run.log`. Nothing here is quoted from the lecture or the brief;
where a published claim is tested, the test is the evidence.

## How to run it

```bash
python run.py              # all five stages
python run.py --quick      # minutes, for a smoke test
python run.py --only 1,2   # selected stages, keeping the others' results
```

Measured on **cpu**, torch 2.12.1+cu130, seed 15. Stages 1 and 2 are numpy
only. Stages 3 to 5 import Session 14's `model.py` and `train.py` unchanged, so this
harness cannot disagree with the model it measures.

## 1. Four published spike rules, scored on one trace

A synthetic grad-norm trace of **300,000 steps** with a warmup, a decaying level,
lognormal noise whose spread falls with the level, and **40 planted spikes** of known
height and position. Same trace for every rule, so the comparison is fair. A spike
counts as found if an alarm lands within 4 steps of it.

**On the grad norm:**

| rule | found | missed | false alarms | false / 10k steps |
|---|---|---|---|---|
| fixed 5.0 | 21 | 19 | 0 | 0.00 |
| OLMo 2 spike score (7 sigma / 1,000) | 40 | 0 | 0 | 0.00 |
| ZClip EMA z (alpha 0.97, z 2.5) | 40 | 0 | 6,057 | 201.90 |
| Megatron ratio to max (10x) | 0 | 40 | 0 | 0.00 |

**On the loss:**

| rule | found | missed | false alarms | false / 10k steps |
|---|---|---|---|---|
| fixed 5.0 | 21 | 19 | 14,882 | 496.07 |
| OLMo 2 spike score (7 sigma / 1,000) | 40 | 0 | 0 | 0.00 |
| ZClip EMA z (alpha 0.97, z 2.5) | 40 | 0 | 2,787 | 92.90 |
| Megatron ratio to max (10x) | 0 | 40 | 0 | 0.00 |

Three results worth stating plainly.

- **The fixed threshold misses 19 of 40.** The grad-norm level decays
  through training, so spikes late in the run pass under a line drawn early. This is the
  usual advice — *do not use a fixed limit* — with a number attached.
- **ZClip's z = 2.5 is not an alert threshold.** It finds every spike, and raises
  **201.9 false alarms per 10,000 steps** — about
  6,057 pages over this run. ZClip *rescales* the
  gradient at that threshold; it never pages anyone. Borrowing the number for an alert
  band is a category error.
- **Megatron's ratio-to-the-maximum rule finds 0 of 40**, on both curves.
  A loss sets its running maximum in the first few thousand steps and never approaches
  10x it again. The rule catches divergence to a huge value, which is what it is for.

So where should the EMA rule sit? Sweeping the threshold on the grad norm:

| z | found | missed | false alarms | false / 10k |
|---|---|---|---|---|
| 2.5 | 40 | 0 | 6,057 | 201.90 |
| 3.0 | 40 | 0 | 1,724 | 57.47 |
| 4.0 | 40 | 0 | 163 | 5.43 |
| 5.0 | 40 | 0 | 13 | 0.43 |
| 6.0 | 40 | 0 | 1 | 0.03 |
| 8.0 | 40 | 0 | 0 | 0.00 |

**z = 5.0** keeps all 40 detections at 0.43 false alarms
per 10,000 steps. The detection rate does not fall until well past that, because
a real spike is many sigma out; all the threshold buys below z = 5 is noise.

The best alarm rule on this trace is **OLMo 2 spike score (7 sigma / 1,000)**: 40 of 40 found,
0 false alarms.

## 2. Downsampling a long run without losing the spike

300,000 steps drawn on a 2000-pixel-wide chart is
150 steps a column. Each method is rasterised at
2000x600 and compared **pixel for pixel** against the full-data
rendering, which is a stronger test than eyeballing the curve.

| method | points | wrong pixels | of | peak value error | note |
|---|---|---|---|---|---|
| mean | 2,000 | 28,475 | 31,259 | 4.7619 | one value a column |
| min/max | 2,000 | 13,110 | 31,259 | 0.0000 | same point budget |
| M4 | 1,992 | 19,452 | 31,259 | 0.0000 | same point budget |
| LTTB | 2,000 | 10,462 | 31,259 | 0.0000 | same point budget |
| min/max per column | 4,000 | 3,019 | 31,259 | 0.0000 | 2 a column |
| M4 per column | 7,953 | 0 | 31,259 | 0.0000 | <=4 a column, as published |

- **M4 is exact: 0 wrong pixels of 31,259.** Keeping the
  minimum, maximum, first and last of each pixel column reproduces the full rendering
  with no error at all, which is what Jugel et al. proved in 2014 (PVLDB 7(10):797-808)
  for two-colour line rasterisation.
- **Min and max alone are not enough**: 3,019 wrong pixels at twice the
  point budget. The first and last values are what close the gap between columns.
- **A mean-only rollup loses the spike**: peak value error 4.76.

One trap, found by getting it wrong first. The rollup's buckets must use **the same
index-to-column mapping the renderer uses to place a point**. Bucketing with
`linspace` while the renderer rounds `i/(n-1)*(w-1)` straddles the boundaries and costs
about 2,700 wrong pixels — which reads as a flaw in M4 rather than a flaw in the
bucketing. The proof is only as good as the alignment.

## 3. The silent checkpoint misload, and how small its only signal gets

A converter writes tensors under the source model's key names; the target wraps its
stack differently, so the names do not match; `load_state_dict(strict=False)` skips them
and reports success, and those parameters keep their random initialization.

Reproduced on Session 14's model. **Every cheap check passes:**

| check | result |
|---|---|
| checkpoint file exists | PASS |
| checkpoint is the right size | PASS |
| the model load returns without error | PASS |
| parameter count looks right | PASS |
| training starts and the loss is finite | PASS |

The state hash is **unchanged by the load** (`state_unchanged_by_load: True`): 88 target parameters kept their random values and
88 checkpoint tensors were written and then ignored.

Now the part that matters. A real converter misnames *one class* of tensor, not all of
them, so the question is how small the signal gets:

| tensors misnamed | parameters left random | step-1 loss | gap vs correct |
|---|---|---|---|
| 100% | 21,538,176 | 5.6052 | +4.4327 |
| 50% | 9,737,856 | 6.2643 | +5.0918 |
| 25% | 5,802,240 | 4.0866 | +2.9140 |
| 10% | 2,114,688 | 1.9704 | +0.7978 |
| 5% | 787,200 | 1.4313 | +0.2588 |
| 2% | 393,600 | 1.2923 | +0.1198 |

Batch-to-batch spread of a *correct* model is **0.0924 nats** (1 sigma). So:

- Below about **5% of tensors misnamed**, the gap is inside the noise. At 2% it is
  +0.1198 against a 2-sigma band of 0.1848 — invisible
  without a reference run.
- **A partial misload is worse than a total one.** 50% misnamed costs
  +5.09 against
  +4.43 for 100%. The correctly
  loaded parts drive confident wrong logits; a fully random model is merely uncertain.
- For reference, the Lightning LM cookbook reports 2.95
  against 2.55 at step one — a
  0.40-nat gap, which sits between our 5% and 10% rows.

Two checks do catch it: a **strict load** (`strict_load_catches: True`) and building the checkpoint from
the **target keyspace** so a missing name is a construction-time error (`target_keyspace_catches: True`).

## 4. Freezing: four constraints that break without an error

**The stale gradient.**
A weight frozen *after* the optimizer was built, still carrying a zero gradient, shrinks
by `1-(1-lr*wd)^20` = **0.1998%** in 20 steps
at lr = 0.001, wd = 0.1. Measured:
**0.1998%** — predicted and observed agree to four digits.

PyTorch skips a parameter whose `grad` is `None`, so `requires_grad_(False)` on its own
is safe. The stale gradient is the entire mechanism: AdamW's decoupled weight decay needs
no gradient, only a parameter in a group. Building the optimizer from trainable
parameters only fixes it — drift 0.0e+00, hash stable
`True`.

The loss cannot see a 0.20% shift in one tensor. **The hash can**:
`hash_catches: True`.

**The grad norm moves when you freeze.**
Over every weight it is **12.1989**; over the trainable ones only,
**0.5579** — a factor of 22. An inherited clip limit
of 1.0 and an alarm band tuned before the freeze both now mean something different.

**Memory, at 16 bytes a trained weight and 2 a frozen one.**

| arrangement | trainable | frozen | MiB | vs nothing frozen |
|---|---|---|---|---|
| experts trained, rest frozen | 56,770,560 | 7,382,400 | 880 | 1.11x |
| experts frozen, rest trained | 7,529,856 | 56,623,104 | 223 | 4.39x |

Which side you freeze decides everything, because the experts hold
88% of the
parameters. This is why QLoRA and Lightning LM both freeze the big side and train a
small one, and why freezing an embedding saves almost nothing.

**Per-part zero-gradient counts**, the number a dashboard watches:

| part | parameters | tensors | gradient elements exactly zero |
|---|---|---|---|
| embedding | 294,912 | 2 | 0.00% |
| attention | 7,077,888 | 24 | 0.00% |
| experts | 56,623,104 | 36 | 12.24% |
| router | 147,456 | 12 | 0.00% |
| norms | 9,600 | 25 | 0.00% |

After 5 real optimizer steps, 51 frozen tensors were re-hashed and
**0 changed**.

## 5. Routing health: the metric Session 14 did not have

Session 14's build measured MaxVio and dead experts. Neither can see a layer whose
balancing bias has taken the routing decision away from its router: the loads come out
even, nothing is dead, and the experts are chosen without reference to the token. The
metric that can see it is the **top-k intervention rate** — the share of positions where
`scores + bias` picks a different set of experts than `scores` alone — read against its
ceiling of `1 - k/N`, together with the **selected top-k mass** read against `k/N`.

`routing.py` implements all five metrics and the two ablations the reference model's team
used as proof: route each token to k random experts, and delete the routed experts
outright. A layer that is really routing loses a great deal of loss under the first.

**The numbers are not in this build yet.** The machine lost its GPU to a kernel upgrade
mid-run, and the CPU fallback had not finished at the time of writing. `python run.py
--only 5` fills this section in; the other four stages are unaffected and complete.

## Files

| file | what it is |
|---|---|
| `run.py` | the five stages; writes `results.json` and `run.log` |
| `detect.py` | the trace generator and the four spike rules |
| `rollup.py` | mean / min-max / M4 / LTTB and a line rasteriser to score them |
| `misload.py` | the silent misload, the five cheap checks, the keyspace-first fix |
| `freeze.py` | optimizer groups, zero-grad counts, grad-norm shift, weight hashes |
| `routing.py` | the five per-layer routing metrics and the two ablations |
| `make_readme.py` | generates this file from `results.json` |
| `results.json` | every number above |
| `run.log` | the training log |
| `stability.ipynb` | a short live version |

`misload.py`, `freeze.py` and `routing.py` import Session 14's `model.py` and `train.py`
unchanged and need `session-14-moe/dense.pt`, which Session 14's `train.py` writes.

## What this does and does not show

- The detector and downsampling results are on **synthetic traces** with planted spikes.
  That is the point — the ground truth has to be known to score a rule — but a real
  grad-norm trace has structure these do not, and a rule that wins here could lose there.
- The misload, freezing and routing results are on a **21M-parameter model**. The
  mechanisms are scale-free; the magnitudes are not.
- Nothing here measures the thing a real run actually fails on, which is hardware. A
  single machine cannot produce a straggler, an Xid, or a silent data corruption.
- The routing stage grows one model one way. It does not sweep gamma, top-k, or expert
  count, so it shows that the intervention rate *can* separate layers, not what a healthy
  value is.

