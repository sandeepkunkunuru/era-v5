# Session 8 — How attention works now, in the order it was invented

**Live app → [era-v5.netlify.app/session-8/](https://era-v5.netlify.app/session-8/)**

A chronological account of thirty attention mechanisms, 2014 → 2026. Standard scaled dot-product
attention is built first, from six tokens and four dimensions with every number computed live;
everything after it is presented as an answer to a bill the previous work left unpaid, with honest
pros *and* cons and a straight answer to "when would I actually pick this?"

Ordered by **launch date**, not by teaching order and not grouped by family — which is the point.

---

## What's here

```
session-8/
  index.html     the app — layout, the live attention walkthrough, the two-bills chart, the timeline
  data.js        the thirty mechanisms: date, lane, problem, mechanism, buys, costs, pick-when, source
  README.md      this file — the source table below is the authority for every date in the app
```

No build step, no dependencies, no external JS. Open `index.html`, or:

```bash
python3 -m http.server 8000     # then visit /session-8/
```

---

## Sources for the chronology

**Every date below is a v1 submission date read from the arXiv abstract page, or the primary release
announcement where there is no paper.** None of them is from recall. This matters because the
assignment brief is explicit about it, and because three widely-repeated dates turn out to be wrong
(see *Corrections* below).

Where a mechanism has a famous later paper, the row is its **first public appearance**, and the later
work gets its own row rather than being folded in.

| # | Date | Mechanism | Authors | Primary source | Lane |
|--:|---|---|---|---|---|
| 1 | 2014-09-01 | Attention (additive) | Bahdanau, Cho & Bengio | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) | mechanism |
| 2 | 2017-05-08 | Learned absolute positions | Gehring, Auli, Grangier, Yarats & Dauphin (ConvS2S) | [arXiv:1705.03122](https://arxiv.org/abs/1705.03122) | position |
| 3 | 2017-06-12 | Scaled dot-product attention + multi-head | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | mechanism |
| 4 | 2017-06-12 | Sinusoidal positions | Vaswani et al., §3.5 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | position |
| 5 | 2019-01-09 | Transformer-XL (segment recurrence) | Dai, Yang, Yang, Carbonell, Le & Salakhutdinov | [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) | memory |
| 6 | 2019-04-23 | Sparse Transformer | Child, Gray, Radford & Sutskever (OpenAI) | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) | compute |
| 7 | 2019-11-06 | Multi-Query Attention (MQA) | Shazeer | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | memory |
| 8 | 2020-04-10 | Sliding window attention | Beltagy, Peters & Cohan (Longformer) | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) | compute |
| 9 | 2020-06-29 | Linear attention | Katharopoulos, Vyas, Pappas & Fleuret | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) | compute |
| 10 | 2021-02-22 | The delta rule (DeltaNet) | Schlag, Irie & Schmidhuber | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) | memory |
| 11 | 2021-04-20 | RoPE | Su, Lu, Pan, Murtadha, Wen & Liu (RoFormer) | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | position |
| 12 | 2021-08-27 | ALiBi | Press, Smith & Lewis | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | position |
| 13 | 2022-05-27 | **FlashAttention** | Dao, Fu, Ermon, Rudra & Ré | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | compute |
| 14 | 2023-05-22 | GQA | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón & Sanghai | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | memory |
| 15 | 2023-06-28 | NTK-aware scaled RoPE | bloc97 | [r/LocalLLaMA `14lz7j5`](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) | position |
| 16 | 2023-07-17 | RetNet (retention) | Sun, Dong, Huang, Ma, Xia, Xue, Wang & Wei | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) | compute |
| 17 | 2023-08-31 | YaRN | Peng, Quesnelle, Fan & Shippole | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | position |
| 18 | 2023-09-29 | Attention sinks / StreamingLLM | Xiao, Tian, Chen, Han & Lewis | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | memory |
| 19 | 2023-10-03 | Ring Attention | Liu, Zaharia & Abbeel | [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) | compute |
| 20 | 2023-12-01 | Mamba (selective SSM) | Gu & Dao | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | compute |
| 21 | 2024-04-30 | Multi-token prediction | Gloeckle, Youbi Idrissi, Rozière, Lopez-Paz & Synnaeve | [arXiv:2404.19737](https://arxiv.org/abs/2404.19737) | mechanism |
| 22 | 2024-05-07 | MLA (multi-head latent attention) | DeepSeek-AI (DeepSeek-V2) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | memory |
| 23 | 2024-06-10 | DeltaNet, parallelised | Yang, Wang, Zhang, Shen & Kim | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | memory |
| 24 | 2024-10-07 | Differential Transformer | Ye, Dong, Xia, Sun, Zhu, Huang & Wei | [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) | mechanism |
| 25 | 2024-12-09 | Gated DeltaNet | Yang, Kautz & Hatamizadeh (NVIDIA) | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | memory |
| 26 | 2025-01-14 | Lightning attention at scale | MiniMax (MiniMax-01) | [arXiv:2501.08313](https://arxiv.org/abs/2501.08313) | compute |
| 27 | 2025-02-16 | NSA (native sparse attention) | Yuan et al. (DeepSeek) | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | compute |
| 28 | 2025-09-11 | Hybrid depth schedule (3 : 1) | Qwen Team | [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | memory |
| 29 | 2025-09-29 | DSA + lightning indexer | DeepSeek-AI (V3.2-Exp) | [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) | compute |
| 30 | 2025-12-13 | DroPE | Gelberg, Eguchi, Akiba & Cetin (Sakana AI) | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) | position |

Secondary sources used for figures quoted in the app, not for dates:

- DeepSeek-V3 training cost (2.788M H800 GPU-hours ≈ $5.576M) — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- The depth/width result behind LightningLM V4's 20-layer 120B — Fahim & Karim, *The Depth Delusion*, [arXiv:2601.20994](https://arxiv.org/abs/2601.20994), 2026-01-28
- LightningLM V4's DDDGDDDG motif, Memory Stream and DroPE application — the V4 cookbook, [arXiv:2606.07404](https://arxiv.org/abs/2606.07404)

### Corrections this exercise turned up

Three dates that are routinely stated wrong, including in class:

1. **The delta rule is 2021, not 2024.** Schlag, Irie & Schmidhuber introduced it in
   [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) (22 Feb 2021). The 2024 paper
   ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484)) *parallelised* it; **Gated** DeltaNet is a
   third paper ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464), 9 Dec 2024). Three papers
   across three years, usually collapsed into one. The V4 cookbook cites Schlag (2021) correctly.
2. **Sparse attention is 2019 and it is OpenAI's**, not DeepSeek's — Child, Gray, Radford &
   Sutskever, [arXiv:1904.10509](https://arxiv.org/abs/1904.10509). DeepSeek's contribution six years
   later is making it *natively trainable and hardware-aligned*, which is a stronger claim than
   inventing it.
3. **Attention is 2014, not 2017.** Bahdanau, Cho & Bengio,
   [arXiv:1409.0473](https://arxiv.org/abs/1409.0473). The 2017 paper removed the recurrence; it did
   not invent the mechanism.

---

## Q2 — what the timeline shows that a list cannot

The full write-up is [§4 of the app](https://era-v5.netlify.app/session-8/#shows). In brief:

1. **Nobody optimised attention for 22 months.** Transformer June 2017; the first paper attacking
   T² is April 2019. Optimisation follows demand, not invention.
2. **The two bills separated within seven months and never merged.** Sparse Transformer (compute,
   Apr 2019) and MQA (memory, Nov 2019) are different authors solving different pain. From that
   point the field runs two threads that only rejoin in 2025's hybrid schedules — invisible if you
   group by family.
3. **Good ideas arrive years before they are usable, and a kernel is what unlocks them.** The delta
   rule is Feb 2021 and ships in 2025. Yang et al. (2024) added *no* modelling idea — they made the
   update run as chunked matmuls instead of a sequential scan. FlashAttention is the same story:
   it changed nothing about what attention computes and everything about what it costs.
4. **The field changed its mind about exactness twice.** Approximate (2019–2021) → FlashAttention
   shows exact was never the expensive part (2022) → approximation returns (2024–2025), but *trained
   in* rather than bolted on.
5. **Position ends by being deleted.** store → compute → rescale → remove. Each step stores less,
   and DroPE's claim is that positional encoding helps convergence and then blocks generalisation.
6. **The most-adopted context-extension trick of 2023 was a Reddit post.** NTK-aware scaled RoPE
   shipped in production models before YaRN formalised parts of it. Build the timeline from arXiv
   alone and you get the wrong story.
7. **Compromises beat extremes, and arrive late.** MQA 2019 → GQA 2023. Four years for "two instead
   of one or eight", adopted near-universally within a year — because GQA shipped an *uptraining
   recipe*, not just a better idea. Adoption is an engineering property.
8. **The answer stopped being a mechanism and became a schedule.** Qwen3-Next runs 3 Gated DeltaNet
   layers to 1 attention layer; LightningLM V4's DDDGDDDG is 6 DeltaNet to 2 GSA — **the same 3:1
   ratio, reached independently.**

**And the prediction it licenses:** each wave takes something bolted on at inference time and makes
it part of training — sparsity did this in 2025 (NSA), position in 2025 (DroPE), the cache in 2024
(MLA compresses by construction rather than by eviction). The remaining hand-set component is the
**schedule**: every hybrid ratio in existence, both 3:1 results included, was chosen by hand. A
learned depth schedule is the obvious next entry, and nothing on this timeline has done it.

### The mechanism that wasn't covered

**FlashAttention** — Dao, Fu, Ermon, Rudra & Ré, **27 May 2022**,
[arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

Every other mechanism on the required list buys cheapness with approximation. FlashAttention buys it
with nothing — the output is bit-for-bit identical to standard attention, 2–4× faster, O(T) memory
instead of O(T²), and there is no quality column to report. It matters *to the timeline* because it
shows the preceding three years had misdiagnosed the bottleneck: the enemy was never arithmetic, it
was writing the T×T matrix to HBM and reading it back.

Leave it out and the timeline tells a clean, wrong story — attention was expensive, so the field
approximated it and got better at approximating. Put it in and the true one appears: the field
approximated attention for three years on a mispriced assumption, someone repriced it, the
approximation work paused, and what returned was a more careful kind of approximation.

Five more uncovered mechanisms are in the timeline with verified sources: **Transformer-XL**
(2019-01-09) — the direct ancestor of V4's own Memory Stream; **Ring Attention** (2023-10-03);
**RetNet** (2023-07-17); **Mamba** (2023-12-01); and **Differential Transformer** (2024-10-07), the
one entry that treats attention as *wrong* rather than expensive.

---

## Design notes

- **Four lanes, three hues plus a neutral** — compute (teal), memory (coral), position (gold), and
  the mechanism itself (grey). The trio was checked with the data-viz palette validator against this
  page's surface `#0b1020`: worst all-pairs CVD separation ΔE 8.4 (deutan, target ≥ 8), worst
  normal-vision ΔE 22.7 (floor ≥ 15), all three ≥ 3:1 contrast. Every lane also carries its **name as
  text** on each card, so identity is never colour-alone.
- **The attention heatmap uses one hue, light→dark**, because it encodes magnitude rather than
  identity. Label ink is picked from the ramp *step*, so all five steps clear WCAG AA (4.5:1)
  against their own cell — verified numerically, not by eye.
- **The two-bills chart is single-axis.** Compute and memory are different quantities, so they are
  both indexed to T = 1,024 → 1× and plotted on one log scale rather than on two y-axes.
- The `data.js` entries are the single source of truth: the timeline, the lane counts and the source
  table at the bottom of the page are all generated from them, so they cannot drift apart.

---

*ERA V5 · Session 8 assignment. Lecture report (with 16 corrections against the recording and the V4
cookbook) lives in the private course repo.*
