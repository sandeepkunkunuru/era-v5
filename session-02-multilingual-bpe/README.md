# ERA V5 · Session 2 — Multilingual BPE Tokenizer (India Wikipedia)

Joint 10k-vocab BPE for English/Hindi/Telugu/Spanish that minimizes the **cross-lingual fertility gap**
`X4 − X1` (fertility `Xl = tokens/word`) subject to **English ≤ 1.2**. Score = `1000/(X4 − X1)`.

## Run
```bash
~/projects/venv/bin/python fetch_corpora.py   # (re)fetch the 4 India Wikipedia pages into data/
~/projects/venv/bin/python train.py           # search allocation weights, save best tokenizer + results
```

## Outputs
- `tokenizer.json` — the trained tokenizer (graders re-run this).
- `tokens.txt` — the 10,000-token vocab (downloadable in the widget).
- `results.json` — fertilities, gap, score, weights, word counts.
- `results_all.json` — the full search (allocation frontier).

## Current best (feasible, en ≤ 1.2)
weights `en20/hi8/te16/es8` → en 1.142 / hi 1.471 / te 1.949 / es 1.488, gap 0.807, **score ≈ 1,239**.
See `../../llm-cloud/era-v5/session-02-multilingual-bpe/HANDOFF.md` for the full analysis and next steps
(script-disjoint allocation to improve the score).

## Key design choices
- **char/unicode-level** BPE, not byte-level (byte-level triples Indic fertility).
- **`WhitespaceSplit`** pre-tokenizer (whitespace-only) to match the grader's `\S+` word count.
