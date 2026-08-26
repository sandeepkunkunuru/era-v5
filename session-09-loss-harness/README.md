# ERA V5 · Session 9 — the loss harness

Everything between the model's output and the scalar the optimiser sees, made **observable**,
and then a second head predicting `t+2`.

Every number on this page came out of `loss_harness.py`. `run.log` is the transcript of the
run that produced them and `results.json` is the same numbers machine-readable; this file is
generated from that JSON by `make_readme.py`, so it cannot drift from the run.

```bash
python loss_harness.py          # the full run: writes run.log and results.json
python loss_harness.py --quick  # same structure, 200 training steps
python make_notebook.py         # rebuild loss_harness.ipynb
```

**Reproducing it:** seed `20260822` throughout, tiny Shakespeare via GPT-2 BPE
(`V = 50,257`), torch 2.12.1+cu130 on NVIDIA GeForce RTX 4050 Laptop GPU. The notebook
[`loss_harness.ipynb`](loss_harness.ipynb) runs the same functions cell by cell on a Colab T4.

| file | what |
|---|---|
| [`loss_harness.py`](loss_harness.py) | the harness — one function per requirement |
| [`model.py`](model.py) | a plain pre-norm transformer: RMSNorm, SwiGLU, tied-or-untied head, optional `t+2` head |
| [`loss_harness.ipynb`](loss_harness.ipynb) | the Colab version |
| [`run.log`](run.log) | full transcript of the run below |
| [`results.json`](results.json) | every number, machine-readable |

---

## Part 1 — the seven numbers

### 1. Every tensor shape, and what each dimension means

| tensor | shape | what the dimensions are |
|---|---|---|
| `tokens` | `(4, 256)` | B sequences × T positions; each entry is one token id |
| `hidden` | `(4, 256, 384)` | B × T × D; one D-vector per position, having seen its whole left context |
| `lm_head.weight` | `(50257, 384)` | V rows × D; one learned row per vocabulary token |
| `logits` | `(4, 256, 50257)` | B × T × V; one raw score per vocabulary token, per position |
| `inputs` | `(4, 255)` | B × (T−1); the last position is dropped — nothing follows it |
| `targets` | `(4, 255)` | B × (T−1); the first token is dropped — nothing predicts it |
| `flat logits` | `(1020, 50257)` | (B·(T−1)) × V; `cross_entropy` wants one row per prediction |
| `flat targets` | `(1020,)` | (B·(T−1)); one correct token id per prediction |

**The logits tensor is 131× larger than the hidden state that produced it.** That ratio is
exactly `V/D` — 50,257 / 384 here — so it does not depend on batch or context at all.
At V5's shape (`V` = 131,072, `d_model` = 4,096) the same ratio is **32×**. Either way it is a
constant multiplier sitting on the last layer, which is the whole of §8 of the session.

### 2. The shift, verified in strings

`assert inputs[1:] == targets[:-1]` — asserted, not eyeballed. Printed as text rather than ids,
because an off-by-one is invisible in a wall of integers:

```
inputs : 'First Citizen:\nBefore we proceed any further, hear me'
targets: ' Citizen:\nBefore we proceed any further, hear me speak'
```

The harness also prints the bug the lecture warns about twice — targets un-shifted, so the model
is handed its own input as the answer. The loss curve that produces is beautiful and meaningless.

### 3. Padding, and the contributing-token count

Four sequences of lengths [64, 48, 28, 16], padded to a common width.

| | value |
|---|--:|
| positions if padding counts | **252** |
| positions after masking | **152** (100 pad targets dropped, 40% of the batch) |
| loss counting padding | 9.2286 |
| loss ignoring padding | 4.2502 |
| difference | -4.9784 |

The padded loss is the flattering one: padding is trivially predictable, so once the model has
learned `<pad>` those positions cost almost nothing and drag the mean down.

**The denominator, separately.** Even with an `ignore_index`, dividing by `B·T` instead of the
real count rescales the loss: 4.2502 becomes
**2.5636**. That factor is whatever fraction of the batch was real,
so it moves with every batch — the loss curve becomes partly a plot of your padding ratio.

**`<eos>` is not `<pad>`.** The lecture masks both. `<pad>` is an artefact of batching and must
go; `<eos>` is a real token the model has to learn to emit, and masking it means the model never
learns to stop generating.

**The trap, demonstrated.** Note which way the loss moved above: this model has never been trained
on padded batches, so it gives `<pad>` almost no probability and those positions are the
*expensive* ones — masking them **lowers** the number. The famous trap is the opposite case, and
it needs a model that has *learned* `<pad>`. So we trained a copy on padded batches for 300 steps
and measured again:

| | counting pad | masking pad |
|---|--:|--:|
| before padded training | 9.2286 | 4.2502 |
| after padded training | **2.5965** | 4.0390 |
| fell by | **6.6321** | 0.2113 |

**Only 3% of that apparent progress is real.** The counted loss dropped
6.63 nats and the honest loss dropped 0.21 — the remainder is the model
getting good at predicting padding, which is worth exactly nothing at inference. This is the
lecture's *"loss will drop tremendously and I will become happy"* `[01:27:54]`, as a number.

### 4. Two documents packed into one sequence

Shakespeare and Python source, alternating, so the join is as unrelated as real packing makes it:
256 tokens, 7 boundaries (2.7% of positions).

| | value |
|---|--:|
| loss without the boundary mask | 6.8536 |
| loss with the boundary mask | 6.5536 |
| difference | -0.3000 |
| mean loss **at** the 7 boundary positions | **17.4826** |
| mean loss at the other positions | 6.5536 |
| a boundary position costs | **2.67×** an ordinary one |

**Explanation.** A boundary position asks the model to predict the first token of a Python file
from the last token of a Shakespeare speech. There is no relationship to learn, so the loss there
is noise — and training on it actively teaches the model that unrelated things follow each other.

**Explanation.** Nothing connects the two documents, so at a boundary the model cannot do better
than its prior — which is why those positions are the **most expensive in the sequence**,
2.67× an ordinary one.

So masking them makes the reported loss **fall**. That is not the loss being flattered. The
positions removed were the ones teaching the model something false — that a Python file follows a
Shakespeare speech — and the *unmasked* number is the dishonest one: an average that includes a
question with no answer, which drifts with your packing density rather than with your model.

Note this only becomes visible on a **trained** model. On an untrained one every position costs
`ln(V)`, boundary or not, and the two numbers agree to four decimals — so a harness that measures
this before training measures nothing. The Part 1 model is given 400
steps of ordinary next-token training first, for exactly this reason.

### 5. Perplexity, and where an untrained model starts

| | value |
|---|--:|
| vocabulary `V` | 50,257 |
| `ln(V)` — the uniform-guess loss | **10.8249** |
| measured loss, untrained | **10.9289** |
| measured perplexity | **55,764** |
| perplexity / V | 1.110 |
| excess over `ln(V)` | +0.1040 |

The lecture's rule — *"if your loss does not start between 12 and 11, something is wrong, do not
train"* — is right, and the run confirms it. But the anchor is a **floor, not a target**, and the
sign of the deviation tells you which thing is wrong.

`ln(V)` is the loss of a model that is exactly uniform. Any spread in the initial logits is
confidence the model has not earned, and against a random target unearned confidence costs more
than it saves — so an untrained model lands *above* `ln(V)`, never below. **Below `ln(V)` means
leakage** (a target-alignment bug, or the answer visible in the input). **Far above** means the
head is initialised too hot. The +0.1040 excess here is a free measurement
of how opinionated the initialisation is.

### 6. Tied against untied

| | value |
|---|--:|
| `V × D`, the head matrix | 50,257 × 384 = **19,298,688** |
| total parameters, **tied** | 30,018,816 |
| total parameters, **untied** | 49,317,504 |
| cost of untying | **19,298,688** (+64.3%) |

At V5 scale the same arithmetic stops being forgiving: `131,072 × 4,096` = **536.9M** parameters
(1.00 GiB in bf16 before optimiser state), and at `d_model` 8,192 it is **1.07B**.

And Session 7 closes the escape. With a byte-codec input side there is no `[V,D]` input table to
tie *to*, so "just tie it" is not available to V5 at any price — which is why §23 of the session
lists a factored head as an open question rather than a solved one.

### 7. Peak memory — ordinary cross-entropy against a chunked one

The chunked version computes `chunk` rows of logits at a time inside a checkpoint, so they are
freed after the forward and recomputed during the backward: arithmetic traded for memory. The
loss must not move, and it does not.

**At this model's head** — N=4,096 positions, D=384, V=50,257, bf16:

| implementation | peak MiB | loss | vs naive |
|---|--:|--:|--:|
| naive (all logits at once) | 2,650.5 | 10.910684586 | 1.00× |
| chunked, chunk=1024 | 958.0 | 10.910684586 | 2.77× |
| chunked, chunk=256 | 516.3 | 10.910684586 | 5.13× |
| chunked, chunk=64 | 442.7 | 10.910687447 | 5.99× |

Largest disagreement between any two losses: **2.86e-06** (fp32
reduction order, not approximation). Floor — one row of logits, i.e. essentially just the head
and its gradient — **366.2 MiB**.

**At the V5 vocabulary** — N=2,048, D=1024, V=131,072, bf16:

| implementation | peak MiB | loss | vs naive |
|---|--:|--:|--:|
| naive (all logits at once) | 3,586.1 | 12.022954941 | 1.00× |
| chunked, chunk=512 | 1,799.1 | 12.022954941 | 1.99× |
| chunked, chunk=128 | 1,543.3 | 12.022954941 | 2.32× |
| chunked, chunk=32 | 1,543.2 | 12.022954941 | 2.32× |

Largest disagreement: **0.00e+00** — bit-identical. Floor
**1,026.4 MiB**.

**The ratio is 5.99× and 2.32× respectively, and it plateaus.** That
plateau is the useful part: chunking removes the *logits*, and what is left — the head matrix,
its gradient, and the copies the matmul needs — it cannot touch. Past a certain chunk size you
are no longer buying anything, which is exactly the sweep the session says V5 has to run
(`[01:48:16]`: a $1,000/hour GPU, or chunk the loss and run cheaper cards slower).

**Projection to the real V5 shape** (arithmetic from the shapes, not measured on a 6 GiB laptop
card):

| batch | context | logits, bf16 | with the backward |
|--:|--:|--:|--:|
| 8 | 8,192 | 16.00 GiB | 32.00 GiB |
| 4 | 32,768 | 32.00 GiB | 64.00 GiB |
| 1 | 262,144 | 64.00 GiB | 128.00 GiB |

The last row is one intermediate tensor, larger than any accelerator sold, for a quantity whose
entire purpose is to be collapsed into a single scalar.

---

## Part 2 — a second head at `t+2`

Same trunk, two heads, losses added. 1500 steps; the `t+2` head adds
12,865,792 parameters.

| | head 1 (`t+1`) | head 2 (`t+2`) | sum | gap `L2−L1` |
|---|--:|--:|--:|--:|
| step 0 | 10.8619 | 10.8681 | 21.7300 | +0.0062 |
| step 1500 | **3.7687** | **4.4709** | **8.2396** | **+0.7021** |
| change | -7.0932 | -6.3972 | -13.4904 | |

**Both heads start at `ln(V)` = 10.8249, and they have to.** At step 0 neither knows anything,
and "the token after next" is exactly as unknown as "the next token". The gap is created by
training, not by initialisation.

**By the end the ordering is the expected one and the gap widens** as `L1` falls. That is the
point rather than a defect: head 2 must predict a token without ever seeing the one in between,
so it forces the hidden state at `t` to carry information the next-token loss alone would never
ask for. That is the extra supervision MTP buys — and the cost is honest: another full `[V,D]`
matrix, which at V5 scale is another 536.9M parameters per head.

### The thing in the curve that should not be there

For roughly the first fifty steps **the `t+2` head is ahead of the `t+1` head** — the harder
question answered better, by about 0.12 nats. The brief's own widget says the opposite ("head 4's
loss sits above head 1's"). So we tested it instead of narrating past it.

**Hypothesis 1 — tying.** Head 1 shares its matrix with the input embedding, so its gradient is
doing two jobs; head 2 is a free matrix. Prediction: untie head 1 and the crossover disappears.
Control: identical seed, data and steps, `tie_weights=False`.

| step | tied `L2−L1` | untied `L2−L1` |
|--:|--:|--:|
| 0 | +0.0062 | -0.0127 |
| 10 | -0.1358 | -0.1209 |
| 20 | -0.1177 | -0.0991 |
| 30 | -0.1201 | -0.1061 |
| 40 | -0.0574 | -0.0188 |
| 50 | +0.0168 | +0.0246 |
| 75 | +0.1790 | +0.1685 |
| 100 | +0.2480 | +0.2600 |

**Refuted.** The untied arm crosses over at the same steps ([0, 10, 20, 30, 40]) and by
the same margin — slightly *earlier*, if anything. It is a property of the task pair, not of the
parameterisation.

**Hypothesis 2 — the marginal-fitting phase.** Before the trunk carries usable context, `t+1` has
no real advantage over `t+2`, so the crossover should sit where the model stops fitting unigram
frequencies — at the corpus's unigram entropy.

| | value |
|---|--:|
| unigram entropy of the training tokens | **6.3151** nats |
| crossover window | step 40 (`L1`=7.0928) → step 50 (`L1`=6.7028) |
| is the unigram entropy inside that window? | **no** |

**Also refuted.** The crossover happens about
0.58
nats *above* the marginal floor — before the model has finished fitting unigram frequencies at all.

**Two hypotheses, two refutations.** What survives is the observation: reproducible under a fixed
seed, identical in both arms, crossing at the same step. The next discriminating test is a frozen
random trunk — if the effect is about how much context the trunk carries, freezing it should stop
head 1 from ever overtaking. That is not run here and is not claimed either way.

It is worth reporting rather than deleting, because the brief presents "head *k*'s loss sits above
head 1's" as what an MTP loss curve shows, and **for the first fifty steps of a real run it is
simply not true**. Anyone reading an early MTP curve as evidence the extra head is working is
reading a regime nobody has characterised.

---

## What this says about V5

- **The full-vocabulary softmax can stay exact.** Chunking changed peak memory by
  5.99× and the loss by 3e-06. There is no accuracy
  argument for approximating it.
- **The chunk size is a real decision with a plateau.** The ratio stops improving once the head
  matrix and its gradient dominate. That floor is measurable before the cluster is booked.
- **The logits tensor, not the head matrix, is the thing that scales.** `V/D` is a constant
  (32× at V5's shape, 131× at this harness's); `B·T` is not. At 256K context that constant
  is multiplying enough positions to make one intermediate tensor 64 GiB.
- **Untying costs +64.3% here and there is no tying option at V5.** Which makes
  "factored head or pay 536.9M" the open question the session ends on, and the MTP head count a
  multiplier on whichever answer wins.
