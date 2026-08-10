# Kronecker v2 · Problem 3 — the dynamic byte window

**ERA V5 · Session 7 assignment.** Of the five open problems, this solves **problem 3**:

> *"Today Kronecker is limiting to presenting 32 position for every word (even 'apple' or 'a' as
> well). That's a waste of space. What can we do? How can it be dynamic and doesn't force us to
> crop a word (currently we cannot have a word of len more than 32)."*
> — extended in the recording: *"what do we do for the indic?"* `[02:27:32]`

```bash
# one-time: fetch the corpus (large + third-party, so not committed)
cd ../session-03-india-first-40b && python fetch_corpus.py && cd -

python run_demo.py      # ~150 s on a laptop GPU; regenerates every number below
```

Needs `torch`, `tokenizers`, `regex`. Runs on CPU; the GPU only makes it faster.
`artifacts/{census,probe,lm,evidence}.json` are committed, so every number below can be
checked without re-running anything.

---

## TL;DR

Replacing the one-hot position factor with a **computed basis** removes the 32-byte cap, cuts the
input path to **a quarter of its size**, and *improves* spelling awareness — verified on a real
131,072-token vocabulary and by training models.

| | v1 `onehot32` | **v2 `hybrid8`** | change |
|---|--:|--:|---|
| Code width | 8,192 | **2,048** | **4× smaller** |
| Trainable input path @ `d_model=8096` | 66,322,432 | **16,580,608** | **−49.7M params** |
| Max token length | **32 bytes** | **unbounded** | cap removed |
| Colliding tokens in the V5 vocab | **1,154** | **0** | fixed |
| Probe accuracy (must distinguish colliding tokens) | **0.500** — chance | **1.000** | fixed |
| Prefix similarity (related − unrelated) | +0.5392 | **+0.5697** | improved |
| LM val loss (identical model, data, seed) | 5.3870 | **5.3808** | −0.0062 |

---

## 1. The diagnosis: it is not a waste problem, it is a collision problem

The assignment frames the 32 slots as *wasted space*. Measuring it on a real vocabulary shows the
waste is the **less serious** of two defects, and the second one is fatal.

I trained a **131,072-token BPE vocabulary** (the V5 number) on the Session 3 India-first corpus —
45 MB across 13 Indian languages plus code and math, the same recipe and corpus as the Session 3
fertility sweep — and encoded every token.

**Truncation** (bytes past the window are dropped) is lossy. **Collision** — two distinct tokens
agreeing on their first 32 bytes, therefore receiving *byte-identical* codes and *identical*
embedding vectors forever — is fatal, and it is silent. Nothing errors.

### The damage at a 32-byte window

```
3,241 tokens truncated (2.47%)      1,154 tokens collided in 460 clusters      92.0% of them Indic
```

| Script | Tokens | Chars in 32B | Truncated | **Collided** |
|---|--:|--:|--:|--:|
| Tamil | 6,909 | 10.7 | 7.76% | **3.39%** |
| Malayalam | 8,333 | 10.7 | 7.16% | **2.96%** |
| Bengali | 13,982 | 10.7 | 3.89% | **1.51%** |
| Devanagari | 12,967 | 10.7 | 3.18% | **0.99%** |
| Kannada | 8,154 | 10.7 | 4.80% | **1.13%** |
| Telugu | 8,652 | 10.7 | 2.79% | **0.69%** |
| **Latin** | 34,354 | **32.0** | **0.11%** | **0.055%** |

> **A Tamil token is 61× more likely to be destroyed than a Latin one.** That is the sovereign risk
> the session named, now with a number attached to it.

### What the collisions actually are

These are not exotic edge cases. They are ordinary inflectional morphology — the grammar of the
languages V5 exists to serve:

| Colliding cluster | Language | What is lost |
|---|---|---|
| `▁பயன்படுத்த` · `▁பயன்படுத்தி` · `▁பயன்படுத்தப்பட்ட` · `▁பயன்படுத்தும்` · `▁பயன்படுத்திய` | Tamil | *to use* / *using* / *was used* / *that uses* / *used* — **five forms of one verb collapsed into a single vector** |
| `▁महाराष्ट्र` · `▁महाराष्ट्रातील` · `▁महाराष्ट्रात` · `▁महाराष्ट्राचे` · `▁महाराष्ट्राच्या` | Marathi | *Maharashtra* + four case forms (in / within / of / of-the) |
| `▁বিশ্ববিদ্যালয়` · `▁বিশ্ববিদ্যালয়ৰ` · `▁বিশ্ববিদ্যালয়ের` · `▁বিশ্ববিদ্যালয়,` | Bengali | *university* and its genitives |
| `▁തിരുവനന്തപുര` · `▁തിരുവനന്തപുരം` · `▁തിരുവനന്തപുരത്തെ` · `▁തിരുവനന്തപുരത്ത്` | Malayalam | Thiruvananthapuram, stem + three inflections |
| `▁ऑस्ट्रेलिया` · `▁ऑस्ट्रेलियन` · `▁ऑस्ट्रेलियाई` · `▁ऑस्ट्रेलियाचा` | Devanagari | *Australia* vs *Australian* — the noun and its adjective |
| `ുമായിരുന്നു` · `ുമായിരുന്നു.` | Malayalam | differs only by a sentence-final period |

*(Every cluster above is copied verbatim from `artifacts/census.json`, not paraphrased.)*

A model using Kronecker v1 **cannot distinguish "was used" from "that uses" in Tamil, or
"Australia" from "Australian" in Hindi.** Not "finds it hard" — cannot, at any scale, for any
amount of compute, because the two inputs are the same vector.

### Widening the window does not fix it

The obvious response — raise `pos_dim` to 48 or 64 — is what the session proposes as the thing to
measure. It is measured here, and it is a bad trade:

| Window | Code width | Truncated | Collided | Indic share |
|--:|--:|--:|--:|--:|
| 32 B | 8,192 | 3,241 | **1,154** | 92% |
| 48 B | 12,288 | 175 | 81 | 42% |
| 64 B | 16,384 | 52 | **30** | 0% |

Doubling the window doubles the projection to **132.6M parameters** and *still* leaves 30 colliding
tokens. It buys an asymptote, not a fix — because any fixed window can be exceeded.

---

## 2. The insight: this is the positional-encoding problem, again

Session 7 §11 diagnoses the absolute learned position table and its hard wall at `max_position`:

> *"there is no signal in the parameters that connects row 4,095 to row 4,096, because they were
> only ever independent rows in a lookup table."*

**The byte-position factor in Kronecker v1 is the same object with the same defect.** `onehot_32(p)`
is a stored one-hot over a bounded index; position 33 has no row, so the byte there is dropped. The
field already solved this for sequence position, and the fix generalises verbatim:

> **Stop *storing* position. Start *computing* it.**

Replace `onehot_P(p)` with a basis function `φ(p)` defined for **every** `p`:

```
v1:  κ(t) = z( (1/√L) Σ_p  onehot₂₅₆(byte_p) ⊗ onehot₃₂(p) ),  L = min(len, 32)
v2:  κ(t) = z( (1/√L) Σ_p  onehot₂₅₆(byte_p) ⊗ φ(p)        ),  L = len          ← no cap
```

Three consequences fall out at once:

1. **No cap.** `φ` is defined for all `p`, so nothing is ever cropped. The 751-byte token in the
   real vocabulary encodes fine.
2. **The width decouples from the window.** `K = dim(φ)` is now a *resolution* choice, not a length
   limit. `K = 8` still represents a 700-byte token — which is what makes the code shrink.
3. **The waste disappears at its root.** v1 spends 32 one-hot slots because it must reserve one per
   position. A computed basis spends `K` dimensions regardless of length, so `a` and `apple` no
   longer each reserve 32 slots.

### The four bases I tested

| Basis | φ(p) | Cap | Smooth in p |
|---|---|---|---|
| `onehot` (v1) | `e_p`, undefined past K | **32 B** | no |
| `fourier` | sinusoids at geometric frequencies | none | **yes** |
| `randproj` | fixed pseudo-random unit vector per p | none | no |
| `relative` | fourier over `u = p/(L−1)` | none | yes |
| **`hybrid`** ← recommended | **half `fourier` ⊕ half `randproj`** | **none** | **yes** |

`hybrid` is the recommendation because the two halves buy different things: the sinusoidal half
keeps shared prefixes close (Kronecker's stated benefit — `train`/`training`/`trainer`), and the
random half supplies near-orthogonality that drives hard pairs apart.

---

## 3. Proof

### 3a. The decisive experiment — a task v1 provably cannot learn

Same shape as the Session 2 position experiment, which showed a token-only model pinned at chance.
Take 24 **real** colliding pairs from the V5 vocabulary and ask the smallest possible question:

```
input  [X]   X is the A-member or the B-member of a colliding pair
target       ans_A if X is the A-member, else ans_B
```

Under v1, `emb(A)` and `emb(B)` are the *same vector* — so the network's input carries zero
information about the label and accuracy is pinned at exactly 50%. The script **asserts the
bit-identity numerically before training**, so this is verified, not argued.

| Arm | Bit-identical pairs | Input-path params | **Accuracy** |
|---|--:|--:|--:|
| `dense_control` | — | 3,200 | 1.000 |
| **`kron_v1_onehot32`** | **24 / 24** | 524,288 | **0.500** ← chance |
| **`dyn_hybrid8`** | **0 / 24** | **131,072** | **1.000** |
| `dyn_fourier16` | 0 / 24 | 262,144 | 1.000 |

**v1 scores exactly chance while using 4× the parameters of the codec that scores 1.000.** This is
falsifiable: if v1 ever exceeded chance here, the argument would be wrong.

### 3b. Is the compression free? — language modelling

Identical transformer (4 layers, `d_model` 256, untied dense head), identical data (662k tokens of
real Indic + English text, tokenized with the same 131k vocabulary), identical seed. **Only the
input path differs.**

| Arm | Input-path params | vs v1 | Val loss | Val ppl |
|---|--:|--:|--:|--:|
| `dense_control` (trainable row per token) | 3,072,256 | 1.46× | 5.5745 | 263.62 |
| `kron_v1_onehot32` | 2,097,152 | 1.00× | 5.3870 | 218.55 |
| `dyn_fourier16` | 1,048,576 | **0.50×** | **5.3642** | **213.62** |
| **`dyn_hybrid8`** | **524,288** | **0.25×** | 5.3808 | 217.20 |

**No regression at a quarter of the parameters** (−0.0062 loss). `fourier16` is the best arm
outright at half the parameters. Both byte codecs beat the dense table here — expected in a
low-data regime, since a byte code shares structure across tokens that a dense table must learn
independently; I would not extrapolate that particular ordering to full scale.

*(Both LM runs, executed independently, produced identical val losses to four decimals — the codecs
are deterministic, as `tests/` asserts.)*

### 3c. Spelling awareness is preserved, not sacrificed

Removing collisions is worthless if it destroys the property that motivates byte codecs. Measured
over 400 token pairs sharing a ≥4-character prefix against 400 random pairs:

| Codec | Related | Unrelated | **Separation** |
|---|--:|--:|--:|
| `v1_onehot32` | +0.7493 | +0.2101 | +0.5392 |
| `dyn_fourier16` | +0.8849 | +0.2800 | **+0.6049** |
| **`dyn_hybrid8`** | +0.8289 | +0.2592 | **+0.5697** |
| `dyn_relative16` | +0.2294 | +0.0749 | **+0.1545** ← collapsed |

The computed bases **improve** on v1. And the negative result matters just as much:

> **Normalised position — the most literal reading of "don't waste 32 slots on 'a'" — is the wrong
> answer.** Rescaling position by token length lets short tokens use the full basis, but it destroys
> prefix structure: `train` and `training` no longer align, because the same letter sits at a
> different `u` in each. Separation collapses from +0.54 to +0.15. **Fixing the waste by rescaling
> costs the very property Kronecker exists to provide.** The fix must come from an *unbounded
> absolute* basis, not a *normalised* one.

---

## 4. What I would put in V5

| Decision | Value | Because |
|---|---|---|
| Input codec | `hybrid`, `K = 8` | zero collisions, no cap, 4× smaller than v1, prefix similarity preserved |
| Code width | 2,048 | `256 × 8` |
| Trainable projection | `2048 × 8096` = **16,580,608** | vs 66.3M for v1, vs 1.06B dense — **64× smaller than a dense table** |
| `pos_dim` | **retired as a concept** | there is no window to tune; the answer to "32 or 64?" is "neither" |

The `embedding_policy_id` record (report §2i) gains `basis_kind`, `K`, and `basis_seed`; the codec
is still frozen by construction, so the tokenizer-hash argument is unchanged — and now `pos_dim`
never has to be re-litigated when a new script is added.

---

## 5. Honest limits

- **Scale.** The LM runs are `d_model=256`, 4 layers, 662k tokens, 12k-token working vocabulary.
  This shows the codec is *not worse* and is far cheaper; it does **not** establish behaviour at
  40B. The instructor's own caveat applies to my result as much as his: Kronecker has never been
  validated above 131M parameters.
- **The 1/16 headline is v1's, not mine.** My comparison is against v1 at equal `d_model`.
- **`randproj` is seed-dependent.** The seed must be recorded in the policy id or the codec is not
  reproducible. `tests/` asserts determinism given a fixed seed.
- **Margin, not orthogonality.** The dynamic codec separates the hard pairs by a *small* cosine
  distance (~0.002 worst case). It is nonzero and consistent — which is what the projection needs,
  and the probe confirms it is sufficient — but it is not a large margin, and at very long token
  lengths the fourier half's low frequencies will crowd. `hybrid`'s random half is what keeps this
  from degrading; a pure-fourier codec would be the more fragile choice.
- **Corpus.** 45 MB is thin for a 131k vocabulary, so absolute collision counts would shift on a
  production corpus. The *asymmetry* between scripts is structural (3 bytes/char vs 1) and would not.

---

## 6. Layout

```
run_demo.py               one command; regenerates every number above
dyn/build_vocab.py        trains the 131,072 BPE vocabulary (Session 3 corpus)
dyn/codecs.py             v1 onehot + fourier / randproj / relative / hybrid
dyn/census.py             per-script truncation + collision counts, separation, prefix similarity
dyn/probe.py              the decisive experiment
dyn/model.py              tiny transformer; pluggable dense | Kronecker input path
dyn/train.py              the language-modelling comparison
dyn/scripts.py            Unicode script identification
tests/test_invariants.py  13 tests — determinism, cropping, collision, cost, prefix structure
artifacts/                census.json · probe.json · lm.json · evidence.json
```

`run_demo.py` ends with a 12-row evidence table, each row a claim from this README with its
measured value and PASS/FAIL. Current state: **12/12 PASS**.
