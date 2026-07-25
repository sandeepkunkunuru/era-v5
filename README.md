# ERA v5 — The School of AI

Interactive proofs for the **ERA v5** course. Each session ships as a page where the
model trains **live in your browser** (TensorFlow.js) — no slides, no precomputed
screenshots. You press a button, the boundary moves, and you check the claim yourself.

**Live site:** https://era-v5.netlify.app

## Sessions

| # | Title | What it proves |
|---|-------|----------------|
| [Session 1](./session-1) | Four proofs | activations · depth · embeddings · data — see below |
| [Session 2](./session-2) | Multilingual BPE | one 10k-vocab tokenizer made *fair* across four scripts — parity-aware BPE, score 2,511 |
| [Session 3](./session-3) | India-first 40B data design | vocab size derived from a *measured* fertility sweep, not a round number |
| [Session 4](./session-4) | Data cleaning & dedup | the 8-stage pipeline run for real on 69.4M tokens — MinHash+LSH & decontam from scratch |
| [Session 5](./session-5) | Data mixtures & curriculum | mixture-and-curriculum plan for V5 *(in progress)* |

### Session 2 — cross-lingual fertility parity

One BPE tokenizer, 10,000-token joint vocabulary, tuned so tokens-per-word is as equal as
possible across **English · Hindi · Telugu · Spanish**, with English held under 1.2. The score is
`1000 / (X₄ − X₁)` — smaller spread, higher score.

| | Result |
|---|---|
| **The method** | **parity-aware BPE** (Foroutan et al., ACL 2026): at every merge, merge the *worst-compressed* language's most valuable pair — not the globally most frequent one. Fairness optimized during training. |
| **Fairness ceiling** | with no cap, all four converge to an identical **1.39** tokens/word — gap **0**. The English ≤1.2 cap is what creates the gap (English → 1.18, the other three → an identical 1.58). |
| **Result** | naive joint BPE starves Telugu to ~1.95 (score 1,239); parity-aware BPE → **score 2,511**. Verified exact against HuggingFace `tokenizers`. |
| **Honesty** | the page re-tokenizes all four pages live in-browser (a faithful BPE port) and checks against the reported numbers |

Trained in `session-02-multilingual-bpe/`. Progression: `train.py` (naive, 1,239) → `train_h3.py`
(script-disjoint union, 2,336) → `train_h4.py` (+ whole-word reclamation, 2,430) → **`train_h5.py`
(parity-aware BPE, 2,511)**. `experiments/parity_bpe.py` shows the pure-parity (gap 0) result;
`build_widget.py` publishes artifacts into `session-2/`. Prior-art catalog:
`ai_research/topics/01-tokenization/`.

### Session 3 — India-first 40B, data designed backward from fertility

A terse design report for the data behind a 40B India-first model: pretraining / post-training / RL /
alignment sourcing, India-first cleaning, "Indian-perspective" evaluation, and the tokenizer decision.
The spine is that the **vocabulary size is derived, not guessed** — a measured fertility sweep (32k → 256k)
over English + 12 Indic languages (FLORES-200) + code + math lands the choice at **256k**, anchored against
Gemma 3 (262k), the Tao vocab-scaling law, and Sarvam-1 (68k). Scarcity is stated honestly: verified native
Indic tokens are thin, so synthetic generation is sized, not wished away. Live at
[era-v5.netlify.app/session-3](https://era-v5.netlify.app/session-3/); built in `session-03-india-first-40b/`.

### Session 4 — data cleaning & deduplication, run for real

The session's **8-stage pipeline** — extract · normalize · language-ID · quality · MinHash-dedup · PII ·
decontaminate · manifest — applied end to end to a **69.4M-token** slice of `open-thoughts/OpenThoughts-114k`
(8,000 reasoning traces). MinHash+LSH and n-gram decontamination are **implemented from scratch** (numpy),
with decontam checked against GSM8K / MATH-500 / HumanEval test sets. The honest findings are the graded
depth: PII regexes over-fire on math/code and had to be tuned (5,282 → 61 phone matches); decontam came out
at 0 because OpenThoughts was pre-decontaminated; language-ID and dedup are near-no-ops on already-curated
data — reported as 0 rather than hidden.

| | Result |
|---|---|
| **Input → output** | 8,000 → 7,792 rows · 69,352,268 → 66,759,058 tokens (**−3.8%**, cl100k count) |
| **Where the drop comes from** | extract (−1.94M boilerplate tokens) + quality (−207 looping traces) do the work |
| **From scratch** | MinHash (128 perms, 32×4 LSH bands) + union-find clustering; informative-n-gram decontam |

Live at [era-v5.netlify.app/session-4](https://era-v5.netlify.app/session-4/); pipeline in
`session-04-data-cleaning/clean.py` (the 250 MB parquet is gitignored — the run's `stats.json` /
`manifest.json` are committed under `session-4/data/`).

### Session 5 — data mixtures & curriculum *(in progress)*

A defensible **mixture-and-curriculum specification** for V5: a budget share for every capability lane
(general web · code · math · reasoning · agentic · Indic), the Indic split across verified / unverified /
translated / synthetic tiers, a protected always-on floor the data selector may not cross, an anneal reserve
held back for the cooldown, and difficulty / reasoning-length bands — each number defended and staged behind
1B/3B proxy runs. This session's assignment is submitted as **this repository's README**.

### Session 1 — the four claims

| # | Claim | Result |
|---|-------|--------|
| **S1-1** | Activations exist for a reason | linear line ≈ 55% · one ReLU ≈ 100% |
| **S1-2** | Depth without nonlinearity is a lie | 1-layer ≈ 5-linear (a line); ReLU solves; K₁···K₅ → one 2×1 map |
| **S1-3** | Embeddings learn similarity from next-token | same-category tokens cluster (12/12 nearest-neighbours) |
| **S1-4** | Data closes the memorization gap | train/test gap ≈ 15 → 1 pts as data grows 20→2000 |

A shared **random-seed** control regenerates every dataset, so each *Retrain* converges
from fresh data — the numbers wobble, the story doesn't.

## Layout

```
index.html              course hub (links to every session)
shared/css/tokens.css   design system + shared atoms (used by hub + all sessions)
hub/hub.css             hub-specific layout
session-1/
  index.html            hero · four proof consoles · how-it-works
  css/app.css           session-1 layout
  js/
    data.js             seeded synthetic datasets (rings, moons, toy grammar)
    seed.js             shared random seed across the four proofs
    viz.js              canvas helpers (probability field, scatter, bars, embeddings)
    proof1..4.js        one file per claim — builds + trains the models live
    main.js             sizes canvases, wires the seed control + proofs
verify/                 dev-only: headless re-run + puppeteer screenshots
build.sh                stages hub + sessions into dist/ for Netlify
```

## Run locally

```bash
python3 -m http.server 8842        # then open http://localhost:8842
```

## Verify the numbers headless

```bash
cd verify && npm install
node run.mjs                        # re-runs all four proofs, prints the numbers
node shots.mjs                      # puppeteer screenshots (BASE=<url> to target a deploy)
```

## Deploy

Static, no framework. `bash build.sh` stages `dist/`; Netlify publishes it.

```bash
bash build.sh && netlify deploy --dir=dist --prod
```

New session → add `session-N/`, link it from `index.html` and `build.sh`, redeploy.
