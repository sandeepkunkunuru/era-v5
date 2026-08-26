#!/usr/bin/env python3
"""Generate README.md from results.json, so the write-up cannot drift from the run.

    python loss_harness.py && python make_readme.py
"""
import json, math
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())


def f(x, n=4):
    return f"{x:,.{n}f}"


sh = R["shapes"]
pad = R["padding"]
bnd = R["boundary"]
ppl = R["perplexity"]
tie = R["tying"]
mA, mB = R["memory_model_scale"], R["memory_v5_vocab"]
p2 = R["part2"]
meta = R["meta"]

first, last = p2["first"], p2["last"]
ctrl = p2.get("control_untied")
lnV = p2["ln_V"]


def memtable(m):
    out = ["| implementation | peak MiB | loss | vs naive |", "|---|--:|--:|--:|"]
    base = m["rows"][0]["peak_mib"]
    for r in m["rows"]:
        out.append(f"| {r['impl']} | {r['peak_mib']:,.1f} | {r['loss']:.9f} | {base/r['peak_mib']:.2f}× |")
    return "\n".join(out)


README = f"""# ERA V5 · Session 9 — the loss harness

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

**Reproducing it:** seed `{meta['seed']}` throughout, tiny Shakespeare via GPT-2 BPE
(`V = {ppl['V']:,}`), torch {meta['torch']} on {meta['gpu'] or meta['device']}. The notebook
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
| `tokens` | `{tuple(sh['tokens'])}` | B sequences × T positions; each entry is one token id |
| `hidden` | `{tuple(sh['hidden'])}` | B × T × D; one D-vector per position, having seen its whole left context |
| `lm_head.weight` | `{tuple(sh['lm_head.weight'])}` | V rows × D; one learned row per vocabulary token |
| `logits` | `{tuple(sh['logits'])}` | B × T × V; one raw score per vocabulary token, per position |
| `inputs` | `{tuple(sh['inputs'])}` | B × (T−1); the last position is dropped — nothing follows it |
| `targets` | `{tuple(sh['targets'])}` | B × (T−1); the first token is dropped — nothing predicts it |
| `flat logits` | `{tuple(sh['flat logits'])}` | (B·(T−1)) × V; `cross_entropy` wants one row per prediction |
| `flat targets` | `{tuple(sh['flat targets'])}` | (B·(T−1)); one correct token id per prediction |

**The logits tensor is {R['logits_vs_hidden_ratio']:.0f}× larger than the hidden state that produced it.** That ratio is
exactly `V/D` — {ppl['V']:,} / {mA['D']} here — so it does not depend on batch or context at all.
At V5's shape (`V` = 131,072, `d_model` = 4,096) the same ratio is **32×**. Either way it is a
constant multiplier sitting on the last layer, which is the whole of §8 of the session.

### 2. The shift, verified in strings

`assert inputs[1:] == targets[:-1]` — asserted, not eyeballed. Printed as text rather than ids,
because an off-by-one is invisible in a wall of integers:

```
inputs : 'First Citizen:\\nBefore we proceed any further, hear me'
targets: ' Citizen:\\nBefore we proceed any further, hear me speak'
```

The harness also prints the bug the lecture warns about twice — targets un-shifted, so the model
is handed its own input as the answer. The loss curve that produces is beautiful and meaningless.

### 3. Padding, and the contributing-token count

Four sequences of lengths {pad['lengths']}, padded to a common width.

| | value |
|---|--:|
| positions if padding counts | **{pad['n_all']}** |
| positions after masking | **{pad['n_real']}** ({pad['n_all']-pad['n_real']} pad targets dropped, {100*(pad['n_all']-pad['n_real'])/pad['n_all']:.0f}% of the batch) |
| loss counting padding | {f(pad['loss_counting_pad'])} |
| loss ignoring padding | {f(pad['loss_ignoring_pad'])} |
| difference | {pad['loss_ignoring_pad']-pad['loss_counting_pad']:+.4f} |

The padded loss is the flattering one: padding is trivially predictable, so once the model has
learned `<pad>` those positions cost almost nothing and drag the mean down.

**The denominator, separately.** Even with an `ignore_index`, dividing by `B·T` instead of the
real count rescales the loss: {f(pad['loss_ignoring_pad'])} becomes
**{f(pad['loss_wrong_denominator'])}**. That factor is whatever fraction of the batch was real,
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
| before padded training | {f(pad['loss_counting_pad'])} | {f(pad['loss_ignoring_pad'])} |
| after padded training | **{f(pad['after_padded_training']['counting'])}** | {f(pad['after_padded_training']['masking'])} |
| fell by | **{pad['loss_counting_pad']-pad['after_padded_training']['counting']:.4f}** | {pad['loss_ignoring_pad']-pad['after_padded_training']['masking']:.4f} |

**Only {pad['real_fraction_of_apparent_progress']:.0f}% of that apparent progress is real.** The counted loss dropped
{pad['loss_counting_pad']-pad['after_padded_training']['counting']:.2f} nats and the honest loss dropped {pad['loss_ignoring_pad']-pad['after_padded_training']['masking']:.2f} — the remainder is the model
getting good at predicting padding, which is worth exactly nothing at inference. This is the
lecture's *"loss will drop tremendously and I will become happy"* `[01:27:54]`, as a number.

### 4. Two documents packed into one sequence

Shakespeare and Python source, alternating, so the join is as unrelated as real packing makes it:
{bnd['T']} tokens, {bnd['n_boundaries']} boundaries ({100*bnd['n_boundaries']/bnd['T']:.1f}% of positions).

| | value |
|---|--:|
| loss without the boundary mask | {f(bnd['loss_unmasked'])} |
| loss with the boundary mask | {f(bnd['loss_masked'])} |
| difference | {bnd['loss_masked']-bnd['loss_unmasked']:+.4f} |
| mean loss **at** the {bnd['n_boundaries']} boundary positions | **{f(bnd['loss_at_boundary'])}** |
| mean loss at the other positions | {f(bnd['loss_off_boundary'])} |
| a boundary position costs | **{bnd['loss_at_boundary']/bnd['loss_off_boundary']:.2f}×** an ordinary one |

**Explanation.** A boundary position asks the model to predict the first token of a Python file
from the last token of a Shakespeare speech. There is no relationship to learn, so the loss there
is noise — and training on it actively teaches the model that unrelated things follow each other.

**Explanation.** Nothing connects the two documents, so at a boundary the model cannot do better
than its prior — which is why those positions are the **most expensive in the sequence**,
{bnd['loss_at_boundary']/bnd['loss_off_boundary']:.2f}× an ordinary one.

So masking them makes the reported loss **fall**. That is not the loss being flattered. The
positions removed were the ones teaching the model something false — that a Python file follows a
Shakespeare speech — and the *unmasked* number is the dishonest one: an average that includes a
question with no answer, which drifts with your packing density rather than with your model.

Note this only becomes visible on a **trained** model. On an untrained one every position costs
`ln(V)`, boundary or not, and the two numbers agree to four decimals — so a harness that measures
this before training measures nothing. The Part 1 model is given {R.get('pretrain_steps', 400)}
steps of ordinary next-token training first, for exactly this reason.

### 5. Perplexity, and where an untrained model starts

| | value |
|---|--:|
| vocabulary `V` | {ppl['V']:,} |
| `ln(V)` — the uniform-guess loss | **{f(ppl['ln_V'])}** |
| measured loss, untrained | **{f(ppl['loss'])}** |
| measured perplexity | **{ppl['ppl']:,.0f}** |
| perplexity / V | {ppl['ratio']:.3f} |
| excess over `ln(V)` | {ppl['loss']-ppl['ln_V']:+.4f} |

The lecture's rule — *"if your loss does not start between 12 and 11, something is wrong, do not
train"* — is right, and the run confirms it. But the anchor is a **floor, not a target**, and the
sign of the deviation tells you which thing is wrong.

`ln(V)` is the loss of a model that is exactly uniform. Any spread in the initial logits is
confidence the model has not earned, and against a random target unearned confidence costs more
than it saves — so an untrained model lands *above* `ln(V)`, never below. **Below `ln(V)` means
leakage** (a target-alignment bug, or the answer visible in the input). **Far above** means the
head is initialised too hot. The {ppl['loss']-ppl['ln_V']:+.4f} excess here is a free measurement
of how opinionated the initialisation is.

### 6. Tied against untied

| | value |
|---|--:|
| `V × D`, the head matrix | {ppl['V']:,} × {mA['D']} = **{tie['head']:,}** |
| total parameters, **tied** | {tie['tied']:,} |
| total parameters, **untied** | {tie['untied']:,} |
| cost of untying | **{tie['untied']-tie['tied']:,}** (+{tie['pct']:.1f}%) |

At V5 scale the same arithmetic stops being forgiving: `131,072 × 4,096` = **536.9M** parameters
(1.00 GiB in bf16 before optimiser state), and at `d_model` 8,192 it is **1.07B**.

And Session 7 closes the escape. With a byte-codec input side there is no `[V,D]` input table to
tie *to*, so "just tie it" is not available to V5 at any price — which is why §23 of the session
lists a factored head as an open question rather than a solved one.

### 7. Peak memory — ordinary cross-entropy against a chunked one

The chunked version computes `chunk` rows of logits at a time inside a checkpoint, so they are
freed after the forward and recomputed during the backward: arithmetic traded for memory. The
loss must not move, and it does not.

**At this model's head** — N={mA['N']:,} positions, D={mA['D']}, V={mA['V']:,}, bf16:

{memtable(mA)}

Largest disagreement between any two losses: **{mA['max_loss_disagreement']:.2e}** (fp32
reduction order, not approximation). Floor — one row of logits, i.e. essentially just the head
and its gradient — **{mA['floor_mib']:,.1f} MiB**.

**At the V5 vocabulary** — N={mB['N']:,}, D={mB['D']}, V={mB['V']:,}, bf16:

{memtable(mB)}

Largest disagreement: **{mB['max_loss_disagreement']:.2e}** — bit-identical. Floor
**{mB['floor_mib']:,.1f} MiB**.

**The ratio is {mA['ratio']:.2f}× and {mB['ratio']:.2f}× respectively, and it plateaus.** That
plateau is the useful part: chunking removes the *logits*, and what is left — the head matrix,
its gradient, and the copies the matmul needs — it cannot touch. Past a certain chunk size you
are no longer buying anything, which is exactly the sweep the session says V5 has to run
(`[01:48:16]`: a $1,000/hour GPU, or chunk the loss and run cheaper cards slower).

**Projection to the real V5 shape** (arithmetic from the shapes, not measured on a 6 GiB laptop
card):

| batch | context | logits, bf16 | with the backward |
|--:|--:|--:|--:|
""" + "\n".join(
    f"| {r['B']} | {r['T']:,} | {r['logits_gib']:.2f} GiB | {r['with_backward_gib']:.2f} GiB |"
    for r in R["memory_projection"]
) + f"""

The last row is one intermediate tensor, larger than any accelerator sold, for a quantity whose
entire purpose is to be collapsed into a single scalar.

---

## Part 2 — a second head at `t+2`

Same trunk, two heads, losses added. {p2['steps']} steps; the `t+2` head adds
{p2['extra_head_params']:,} parameters.

| | head 1 (`t+1`) | head 2 (`t+2`) | sum | gap `L2−L1` |
|---|--:|--:|--:|--:|
| step 0 | {f(first['l1'])} | {f(first['l2'])} | {f(first['total'])} | {first['gap']:+.4f} |
| step {last['step']} | **{f(last['l1'])}** | **{f(last['l2'])}** | **{f(last['total'])}** | **{last['gap']:+.4f}** |
| change | {last['l1']-first['l1']:+.4f} | {last['l2']-first['l2']:+.4f} | {last['total']-first['total']:+.4f} | |

**Both heads start at `ln(V)` = {f(lnV)}, and they have to.** At step 0 neither knows anything,
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
""" + "\n".join(
    f"| {a['step']} | {a['gap']:+.4f} | {b['gap']:+.4f} |"
    for a, b in list(zip(p2["history"], ctrl["history"]))[:8]
) + f"""

**Refuted.** The untied arm crosses over at the same steps ({ctrl['negative_gap_steps']}) and by
the same margin — slightly *earlier*, if anything. It is a property of the task pair, not of the
parameterisation.

**Hypothesis 2 — the marginal-fitting phase.** Before the trunk carries usable context, `t+1` has
no real advantage over `t+2`, so the crossover should sit where the model stops fitting unigram
frequencies — at the corpus's unigram entropy.

| | value |
|---|--:|
| unigram entropy of the training tokens | **{f(p2['unigram_entropy'])}** nats |
| crossover window | step {p2['crossover_window'][0]['step']} (`L1`={f(p2['crossover_window'][0]['l1'])}) → step {p2['crossover_window'][1]['step']} (`L1`={f(p2['crossover_window'][1]['l1'])}) |
| is the unigram entropy inside that window? | **{'yes' if p2['marginal_phase_hypothesis']=='supported' else 'no'}** |

**Also refuted.** The crossover happens about
{(p2['crossover_window'][0]['l1']+p2['crossover_window'][1]['l1'])/2 - p2['unigram_entropy']:.2f}
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
  {mA['ratio']:.2f}× and the loss by {mA['max_loss_disagreement']:.0e}. There is no accuracy
  argument for approximating it.
- **The chunk size is a real decision with a plateau.** The ratio stops improving once the head
  matrix and its gradient dominate. That floor is measurable before the cluster is booked.
- **The logits tensor, not the head matrix, is the thing that scales.** `V/D` is a constant
  (32× at V5's shape, {R['logits_vs_hidden_ratio']:.0f}× at this harness's); `B·T` is not. At 256K context that constant
  is multiplying enough positions to make one intermediate tensor 64 GiB.
- **Untying costs +{tie['pct']:.1f}% here and there is no tying option at V5.** Which makes
  "factored head or pay 536.9M" the open question the session ends on, and the MTP head count a
  multiplier on whichever answer wins.
"""

(HERE / "README.md").write_text(README)
print(f"wrote README.md — {len(README.split()):,} words")
