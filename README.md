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
