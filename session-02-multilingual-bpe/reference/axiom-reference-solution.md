# ERA V5 Assignment 2 — Reference Solution (published by instructor)

> Captured 2026-07-14 from the Axiom lesson linked in the graded-0 feedback:
> https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmrirwdhc0afw08nmp64282xo/lesson
> This is **instructor-published course material** (third-party), kept here only as the spec our
> resubmission must match. Arithmetic in the Result table was independently re-verified (all exact).

This folder contains a reproducible reference solution for the multilingual BPE tokenizer assignment.

The goal was to build one shared tokenizer with a 10,000-token vocabulary for India Wikipedia pages in:

- English
- Hindi
- Telugu
- **Maithili**

The evaluation corpus here uses a **wiki-faithful Markdown conversion** rather than clipped article
prose. Links, URLs, tables, references, image links, navboxes, and categories are preserved where the
HTML-to-Markdown conversion emits them.

## Result

| Language | Tokens | Faithful Units | Fertility |
|---|--:|--:|--:|
| English  | 111,390 | 186,367 | 0.597692 |
| Hindi    | 51,190  | 88,359  | 0.579341 |
| Telugu   | 24,428  | 36,292  | 0.673096 |
| Maithili | 4,258   | 5,808   | 0.733127 |

- Spread = 0.733127 − 0.579341 = **0.153786**
- Raw score = 1000 / 0.153786 = **6502.56**
- Hindi penalty factor = 1.000000
- Hindi-adjusted score = **6502.56**

All four ratios satisfy the 1.2 threshold under the faithful-unit denominator:

```
English  = 0.597692
Hindi    = 0.579341
Telugu   = 0.673096
Maithili = 0.733127
```

The tokenizer also round-trips punctuation and number separators. For example,
`India's population is 1,428,627,663.` decodes back to the same string.

## Faithfulness Requirement

The submitted tokenizer must preserve visible text:

> `decode(encode(text))` must keep the same **non-whitespace characters** as `text`.

Tokenizers that strip punctuation, brackets, URL characters, apostrophes, number separators, or other
visible symbols are **not acceptable** for this faithful Markdown evaluation. They may produce low token
counts, but those counts are invalid because the tokenizer is not representing the same input.

## Folder Contents (in the downloadable reference package — NOT captured here)

```
tokenizer.json                    trained tokenizer
metrics.json                      saved metrics for this tokenizer
build_wiki_faithful_markdown.py   fetch + convert Wikipedia pages
train_tokenizer.py                train tokenizer from corpus
evaluate_tokenizer.py             evaluate tokenizer.json
corpus/*.faithful.md              generated Markdown corpus snapshots
corpus/*.faithful.txt             same corpus as plain text input
corpus/*.meta.json                corpus metadata
```

## Setup

```
pip install tokenizers regex requests beautifulsoup4 lxml markdownify
```

## Rebuild the Corpus

```
python build_wiki_faithful_markdown.py
```

Fetches Wikipedia REST HTML and writes `corpus/{en,hi,te,mai}.faithful.md`. Corpus counts may change if
Wikipedia pages change; the included snapshots are the ones used to produce the metrics above.

## Train the Tokenizer

```
python train_tokenizer.py
```

Training choices:

- **Model:** HuggingFace BPE
- **Vocab size:** 10,000
- **min_frequency = 1**
- **Normalizer:** NFKC only
- **Pretokenizer:** Metaspace, using `▁` as the space marker
- **Decoder:** Metaspace
- **Training weights:** `{ "en": 3, "hi": 4, "te": 4, "mai": 2 }`

The tokenizer preserves punctuation, brackets, URL characters, apostrophes, number separators, and spaces
through the **Metaspace** pretokenizer/decoder. This matters because the corpus is a faithful Markdown
representation, not a plain word list. **Metaspace is used instead of ByteLevel because ByteLevel spends
too many tokens on UTF-8 bytes for Indic scripts.**

## Evaluate

```
python evaluate_tokenizer.py
```

Score formula:

```
faithful_unit = one contiguous Unicode letter/mark/number run
                OR one visible non-space punctuation/symbol character
fertility(language) = token_count(language) / faithful_unit_count(language)
score = 1000 / (max_fertility - min_fertility)
```

Hindi penalty:

```
hindi_penalty = exp(max(0, hindi_fertility / 1.2 - 1))
hindi_adjusted_score = raw_score / hindi_penalty
```

Since this tokenizer has Hindi below 1.2, its Hindi penalty factor is 1.

## Notes for Students

Do not report numbers from a clipped page or a hidden private corpus. The important part of this
assignment is **reproducibility**. A good submission should include:

- the exact tokenizer file
- code or clear method used to build it
- the exact Wikipedia corpus extraction process
- token counts for all four languages
- fertility ratios
- raw score calculation
- a live widget or notebook that lets the grader inspect/download the tokenizer

If your tokenizer relies on a custom JSON format, include the encoder code. A vocab list without the
actual encoding algorithm is not enough to reproduce your score.
