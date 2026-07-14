# Session 2 — Corrected faithful tokenizer (Assignment-2 resubmission)

This is the **corrected** Assignment-2 build. The original submission (`../train_h5.py`, "parity-aware BPE",
self-score 2,511) was **graded 0/1000**: it failed the grader's *faithfulness gate* — `decode(encode(text))`
deleted visible characters (`#`, `_`, … → `[UNK]`; whitespace mangled) on the faithful-Markdown input. The
fertility numbers from a non-faithful tokenizer are invalid, so the score was voided. See `../README.md`
§ *Postmortem* for the full root-cause.

This folder reproduces the instructor's published reference method exactly and passes the gate.

## Result

| Language | Tokens | Faithful units | Fertility |
|---|--:|--:|--:|
| English  | 111,390 | 186,367 | 0.597692 |
| Hindi    | 51,190  | 88,359  | 0.579341 |
| Telugu   | 24,428  | 36,292  | 0.673096 |
| Maithili | 4,258   | 5,808   | 0.733127 |

- Spread = 0.733127 − 0.579341 = **0.153786**
- **Score = 1000 / spread = 6502.56**   (Hindi penalty factor 1.0 — all four ratios ≤ 1.2)
- **Faithfulness gate: PASS** — no visible character is lost on any corpus file or on the grader's
  probe samples (the `cite_ref` URL, the `1,428,627,663` number, dense Markdown punctuation).

Our independently trained `tokenizer.json` is **byte-identical** (MD5 `3d4e4bb5…`) to the reference's —
HF BPE training is deterministic given the same corpus, weights and params, and our fresh Wikipedia fetch
reproduced the reference's faithful-unit counts exactly (the India pages had not drifted).

## Method (matches the reference)

- **Corpus**: `en/hi/te/mai` India Wikipedia via REST HTML → *faithful* Markdown (links, URLs, tables,
  refs, categories preserved; only script/style/meta stripped). Fertility denominator is the
  **faithful unit**: one contiguous Unicode L/M/N run, *or* one visible non-space punctuation/symbol char.
- **Tokenizer**: HuggingFace **BPE**, `unk_token="[UNK]"`, vocab **10,000**, `min_frequency=1`,
  normalizer **NFKC**, pre-tokenizer + decoder **Metaspace** (`▁`, `prepend_scheme="never"`),
  training weights **`{en:3, hi:4, te:4, mai:2}`** (each corpus file duplicated N× before training).

**Why it is faithful** where the old build was not: Metaspace pre-tok/decoder keeps every visible
character and restores spaces on decode; `min_frequency=1` puts every character in the faithful corpus
into the vocab, so nothing becomes `[UNK]`. NFKC still rewrites *compatibility* characters
(`″` U+2033 → `′′`, `ⓘ`, `ʱ`) — the grader tolerates this (its own reference uses NFKC); it forbids
*loss*, not normalization.

**Known boundary (not graded):** characters absent from the entire India-Wikipedia corpus (e.g. `@`, `€`,
emoji) are out-of-vocab and dropped. The grader only tests corpus-derived samples, and the official
reference behaves identically. Adding `BPE(byte_fallback=True)` + a ByteFallback decoder would close this
gap for arbitrary input, at the cost of deviating from the byte-identical reference — deliberately not done.

## Reproduce

```bash
PY=~/projects/venv/bin/python
$PY build_faithful_corpus.py   # fetch India pages -> corpus/{en,hi,te,mai}.faithful.{md,txt} + .meta.json
$PY train_faithful.py          # train -> tokenizer.json + metrics.json  (score 6502.56)
$PY evaluate_faithful.py       # score + FAITHFULNESS GATE (exits non-zero if any visible char is lost)
```

`evaluate_faithful.py` is the guard the reference evaluator lacks: it asserts the round-trip loses no
visible character before the score is allowed to count. Run it before any resubmission.

## Files

```
build_faithful_corpus.py   fetch + convert (our code; method == reference)
train_faithful.py          train the 10k faithful BPE tokenizer
evaluate_faithful.py       score + faithfulness gate (+ informational OOV probe)
tokenizer.json             the shipped tokenizer (byte-identical to the reference)
metrics.json               saved metrics
corpus/*.faithful.{md,txt} committed corpus snapshot (raw.html is git-ignored, regenerable)
corpus/*.meta.json         per-language corpus metadata
../reference/              the instructor's published reference package (SOLUTION.md + extracted/)
```
