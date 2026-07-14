# ERA V5 · Session 2 — Multilingual BPE Tokenizer

> ## ⚠️ Graded 0/1000 — corrected build is in [`faithful/`](faithful/)
>
> The build described below (`train_h5.py`, "parity-aware BPE", self-score **2,511**) was **graded
> 0/1000**. It failed a precondition we never tested: the grader requires a **faithful** tokenizer —
> `decode(encode(text))` must preserve every visible non-whitespace character of the **faithful-Markdown**
> input. Our tokenizer emitted `[UNK]` for any character absent from the 4 clean pages (`#`, `_`, `` ` ``,
> `*`, …) and mangled whitespace, so its fertility numbers were voided. The corrected, reference-matching
> build lives in **[`faithful/`](faithful/)** and scores **6502.56** with the faithfulness gate passing.
> See the **Postmortem** section below and `reference/axiom-reference-solution.md`.
>
> Three things also changed vs. what this README describes: the 4th language is **Maithili (mai)**, not
> Spanish; the fertility denominator is the **faithful unit** (letter-run *or* single punct/symbol), not
> the whitespace word; and the corpus is **faithful Markdown**, not clipped prose.

---

## Postmortem — why the parity work scored 0

The fairness objective below was sound but solved for the *wrong metric on the wrong precondition*:

1. **Faithfulness is a hard gate, checked first.** `train_h5.py` built its vocab only from characters seen
   in 4 clean Wikipedia pages, with a bare `[UNK]` and no byte fallback, and used a `WhitespaceSplit`
   pre-tokenizer with **no decoder**. Any unseen Markdown character → `[UNK]`; spaces were lost on decode.
   The grader's own failure sample (`https://hi.wikipedia.org/wiki/भारत#cite_ref-1` → `… [UNK] c ite [UNK]
   re f - 1`) reproduces exactly in our shipped `tokenizer.json`.
2. **The fix is Metaspace, not byte-level.** A Metaspace pre-tokenizer + Metaspace decoder preserves all
   visible characters and restores spaces; `min_frequency=1` on the faithful corpus keeps every character
   in-vocab. Byte-level BPE would also be faithful but re-triggers the Indic fertility blow-up.
3. **The metric moved out from under us.** The grader scores `tokens / faithful_unit`, not `tokens / word`,
   on faithful Markdown — so the entire parity-aware apparatus was optimizing a denominator the grader no
   longer uses. Under the real metric a *simple* weighted BPE scores 6502.
4. **Lesson (recorded):** test the acceptance gate before optimizing the objective. A self-reported score
   from an un-round-tripped tokenizer is not a score. `faithful/evaluate_faithful.py` now enforces the gate
   and exits non-zero on any visible-character loss.

---

## Original build (below) — superseded, kept for the record

A single 10,000-token BPE tokenizer for the **"India"** Wikipedia article in **English, Hindi, Telugu,
Spanish**, built so tokens-per-word (**fertility** `Xₗ = tokens/word`) is as *equal* as possible across
languages, with **English ≤ 1.2**. The assignment scores `1000 / (X₄ − X₁)` — smaller spread, higher score.

But the real subject is **fair multilingual tokenization**: making no language pay a "token tax." This
folder is the full record of every attempt, from a naive baseline to the principled SOTA method, plus the
science we uncovered along the way.

**Live widget:** https://era-v5.netlify.app/session-2/ · **Prior-art catalog:** `../../../ai_research/topics/01-tokenization/`

---

## The corpus (`data/`, committed for reproducibility — graders re-run the tokenizer)

| lang | script | page | words (`\S+`) | unique types | chars/word |
|---|---|---|--:|--:|--:|
| en | Latin | India | 10,121 | 3,774 | 6.45 |
| hi | Devanagari | भारत | 8,078 | 2,454 | 5.42 |
| te | Telugu | భారతదేశం | 2,511 | 1,650 | 7.97 |
| es | Latin | India | 10,532 | 3,538 | 5.09 |

`fetch_corpora.py` (re)fetches these via the Wikipedia REST API. **Word = `\S+`** (whitespace split) to match
the grader — so the pre-tokenizer must be `WhitespaceSplit` (whitespace-only), *not* `Whitespace` (which also
splits punctuation and inflates fertility).

---

## Attempts, in order

| # | script | method | en | hi | te | es | gap | score | status |
|---|---|---|--:|--:|--:|--:|--:|--:|---|
| 0 | `train.py` (byte-level, uniform) | byte-level BPE | 1.40 | 3.53 | **5.77** | 1.32 | 4.45 | 225 | ❌ Indic catastrophe |
| 1 | `train.py` | joint **corpus-weighted** BPE | 1.142 | 1.471 | 1.949 | 1.488 | 0.807 | **1,239** | superseded |
| 2 | `train_h3.py` | **script-disjoint union** | 1.180 | 1.608 | 1.577 | 1.607 | 0.428 | **2,336** | superseded |
| 3 | `train_h4.py` | + **whole-word reclamation** | 1.180 | 1.582 | 1.591 | 1.577 | 0.411 | **2,430** | superseded |
| 4 | `train_h5.py` | **parity-aware BPE** (SOTA) | 1.182 | 1.580 | 1.580 | 1.581 | 0.398 | **2,511** | ✅ **shipped** |

Each attempt's saved artifacts are preserved in `baseline_*/`.

### 0. Byte-level BPE — the Indic catastrophe
Byte-level BPE (GPT-2 style) is catastrophic for Brahmic scripts: every Devanagari/Telugu character is 3
UTF-8 bytes, so Telugu explodes to **5.77 tokens/word**. **Lesson:** use **char/unicode-level** BPE. Every
attempt below is char-level.

### 1. `train.py` — joint corpus-weighted BPE → 1,239
One joint 10k BPE; oversample each language's lines by a weight to steer merge allocation; grid-search the
weights `w_en/w_hi/w_te/w_es` to minimize the gap subject to `en ≤ 1.2`. Best: `en20/hi8/te16/es8`.
**Findings:** (a) the natural balance with *no* cap is all four ≈ 1.4 (gap 0.08); (b) the `en ≤ 1.2` cap
forces English below that balance, which *steals* budget and **starves Telugu to 1.95**; (c) English behaves
like a step function (only 3,774 unique words). One joint budget couples every language.

### 2. `train_h3.py` — script-disjoint union → 2,336
Give each **script** its own budget and union the vocabularies: a joint **Latin** group (English + Spanish),
a disjoint **Devanagari** group (Hindi), a disjoint **Telugu** group. Because scripts occupy disjoint
code-point ranges, their merges never fire on each other's text → the union is lossless.
**The subtlety that took a while:** naive *per-language* union fails for **same-script** languages — English
and Spanish are both Latin, so unioning two Latin merge lists lets whichever has priority hijack the other
(measured: English-priority → Spanish 1.95; Spanish-priority → English 2.05). Fix: train English + Spanish
*together* as one Latin group. This independently reinvents **Chung et al. 2020** (language-clustered
vocabularies) and **XLM-V**.

### 3. `train_h4.py` — whole-word budget reclamation → 2,430
Diagnosed that the H3 tokenizer wastes **~17 % of its 10k slots** on intermediate merges that never surface
as final tokens, while whole-word coverage stays low (Hindi 21 %, Spanish 17 %). Reclaim them: train the
union at a *reduced* budget and spend the freed ~650 slots on explicit whole-word merges for the worst
languages. This is the **Picky BPE** (Chizhov 2024) / **BPE-knockout** (Bauwens 2024) idea applied to a
parity objective. Still one valid BPE tokenizer.

### 4. `train_h5.py` — parity-aware BPE → 2,511 (**shipped**)
The principled method (**Foroutan et al., ACL 2026**): change *one thing* in BPE training — at every merge
step, don't take the globally most-frequent pair; take the most valuable pair of the currently
**worst-compressed** language. Fairness is optimized *during* training, not patched on afterward. Run under
the `en ≤ 1.2` cap (with a safe 1.185 margin), the other three settle at an **identical 1.580**. This replaces
the H3/H4 post-hoc pipeline and is what the live widget serves.

---

## Experiments (`experiments/`) — what we learned probing the problem

| file | question | finding |
|---|---|---|
| `optimize.py` | do normalization (NFC/NFD/NFKC) or a finer allocation help? | **No** — normalization is neutral; the ceiling is ~2,400 safe / ~2,600 at the cap-edge (risky). |
| `augment.py` | how far does whole-word reclamation go? | drives H3's 2,336 → ~2,430 (safe) / ~2,600 (cap-edge). |
| `vocap.py` | does a principled VoCap-style allocation beat the grid? | **No** — marginal-utility water-filling lands on the *same* base budget → confirms we're on the allocation frontier. |
| `parity_bpe.py` | the principled parity-aware trainer + is the fertility a floor? | **Pure parity → all four = 1.390, gap 0.** Not a floor: fertility falls toward 1.0 with budget (needs ≥ 11,416 tokens = unique word types; nears 1.0 by V≈20k). Also: **fertility-parity ≠ Rényi-efficiency parity** (Zouhar's info-theoretic quality metric). |
| `scale.py` | does parity survive 12 languages / 5 scripts, off the 4 pages? | **Yes** — spread stays ~0.001 across Latin/Cyrillic/Arabic/Brahmic; no language starved. But **CJK breaks the `\S+`-word metric** (Chinese ~57 chars/"word"). |

---

## Key findings (the science, beyond the score)

1. **Perfect fairness is achievable.** Without the cap, parity-aware BPE drives *all* languages to the same
   fertility (4 langs → 1.390, gap 0; 12 langs → 2.109, spread 0.0014). Standard BPE never finds this because
   it serves whichever language dominates the corpus.
2. **The English cap manufactures the inequity.** Impose `en ≤ 1.2` and English is privileged to ~1.18 while
   everyone else is taxed to ~1.58. The metric rewards you for *mitigating a distortion its own constraint
   creates* — and forcing English low makes total tokenization *more* expensive, not less.
3. **1.390 is not a floor.** It is the value at V=10k; the common fertility falls monotonically with budget
   toward **1.0** (each `\S+` word its own token). Reaching 1.0 needs at least the **11,416 unique word types** in vocab; because BPE also spends slots on intermediate merges, it only *nears* 1.0 in practice (V=20k → 1.05).
   Going below 1.0 needs superword (cross-whitespace) tokens.
4. **Parity is budget-invariant and scale-robust.** Parity-aware BPE holds the spread at ~0.001 at every
   budget and across 12 languages / 5 scripts; low-resource languages are not starved.
5. **Fairness depends on what you equalize.** Equal *fertility* ≠ equal *information*: at fertility ≈ 1.58,
   Telugu carries 16.5 bits/word vs English's 12.1, and Zouhar's Rényi efficiency *inverts* (te 0.70 best,
   es 0.49 worst). And the whole framework assumes whitespace words — it does not apply to CJK/Thai.

---

## Prior art / novelty (gated — see `references.bib`)

After an adversarial web check, **every angle is RED** — the field converged on this exact problem in
2024–2026. Our approaches independently reproduce published work:
- **Parity-aware BPE** — Foroutan et al., ACL 2026 (*our exact H5 method*, has code).
- **Script-disjoint / clustered vocab** — Chung et al. 2020; XLM-V (Liang et al. 2023).
- **Slot reclamation** — Picky BPE (Chizhov 2024); BPE-knockout (Bauwens 2024).
- **Parity-vs-budget frontier** — Arnett et al., NeurIPS 2025.
- **Unigram > BPE for fertility** — Bostrom & Durrett 2020 (the big lever, but the assignment mandates BPE).

Full BibTeX with per-paper reuse verdicts: `../../../ai_research/topics/01-tokenization/references.bib`
(also in Zotero). Honest outcome: a strong understanding check and a clean artifact, **no novel contribution**.

---

## Reproduce

```bash
PY=~/projects/venv/bin/python
$PY fetch_corpora.py     # (re)fetch the 4 India Wikipedia pages into data/
$PY train_h5.py          # train the shipped parity-aware tokenizer -> tokenizer.json / tokens.txt / results.json
$PY build_widget.py      # publish the artifacts into ../session-2/ (the live page reads results.json)

# explore the science
$PY experiments/parity_bpe.py   # pure parity (gap 0) + the info-theoretic V-sweep
$PY experiments/scale.py        # parity across 12 languages / 5 scripts (+ CJK breakdown)
```

Every tokenizer is verified against HuggingFace `tokenizers` and re-computed live in the browser — the
numbers on the widget equal the real tokenizer applied to the real pages.

## Layout

```
fetch_corpora.py     fetch the 4 India Wikipedia pages -> data/
train.py             attempt 1 — joint corpus-weighted BPE (1,239)
train_h3.py          attempt 2 — script-disjoint union (2,336)
train_h4.py          attempt 3 — + whole-word reclamation (2,430)
train_h5.py          attempt 4 — parity-aware BPE, SHIPPED (2,511)
build_widget.py      publish trained artifacts into ../session-2/
experiments/         optimize · augment · vocap · parity_bpe · scale  (see table above)
data/                the 4 committed India pages (en/hi/te/es)
data_scale/          the 14-language corpora for scale.py
baseline_joint_1239/ baseline_h3_2336/ baseline_h4_2430/   preserved artifacts per attempt
tokenizer.json · tokens.txt · results.json   the current (H5) shipped artifacts
```
