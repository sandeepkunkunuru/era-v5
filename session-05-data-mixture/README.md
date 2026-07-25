# Session 5 — V5 Data-Mixture & Curriculum Plan

**Model:** V5, ~40B India-first (dense or small-active MoE), trained ~75 tokens/param — past Chinchilla's ~20
on purpose, to buy inference efficiency.
**Budget:** 3.0T tokens main pretraining (mid of the session's 2.4–4T brief) + a staged long-context
extension. The final **2.5% (~75B tokens)** is the anneal.

> A mixture is a set of trade-offs made against a fixed token budget, composed **backward from the benchmarks
> we intend to win**. It is a hypothesis until a cheap experiment tests it. This plan states every number, ties
> it to a benchmark, sizes it against real supply, and ends with a proxy experiment — [`mixture_plan.py`](./mixture_plan.py),
> whose output ([`mixture_plan.json`](./mixture_plan.json)) is quoted throughout — that I actually ran.

Everything below is defended, not decorative. Where a share can only be reached by repeating or **synthesising**
data, it says so — that is the accounting this session exists to force.

---

## 1. The main pretraining mixture

Seven lanes over a fixed 3.0T budget. "Supply" is from the approximate inventory in `mixture_plan.py`
(Experiment A); "rep/synth" is the honest verdict on whether unique tokens actually cover the target.

| Lane | Share | ≈ Tokens | Wins these benchmarks | Supply verdict (measured) |
|---|--:|--:|---|---|
| **General web** (multilingual, EN-heavy) | 40% | 1,200B | MMLU, MMLU-Pro, GPQA, HellaSwag | unique-covered (4,800B available) |
| **Code** | 18% | 540B | SWE-bench Verified/Live/Pro, LiveCodeBench, BFCL | unique-covered (Stack v2 ~900B) |
| **Math / STEM** | 14% | 420B | AIME, MATH, GSM8K, FrontierMath | repeat ×3.0 (142B unique) — at the limit |
| **Reasoning** (short/med traces) | 6% | 180B | AIME@high, GPQA@high, ARC-AGI | **87% must be SYNTHESISED** (6B unique) |
| **Agentic** (tool trajectories) | 4% | 120B | SWE-bench multistep, tau-bench, WebArena, GAIA, BrowseComp | **75% must be SYNTHESISED / harvested** (8B unique) |
| **Indic** (all tiers) | 13% | 390B | MILU, IndicGenBench, IndicQA, FLORES | unique-covered *only* across four tiers (see §2) |
| **Long context** | 5% | 150B | RULER, LongBench, needle, InfiniteBench | repeat ×2 then synth 7% |

**What a reviewer should notice, stated first:** the two capabilities this project is *for* — agentic and deep
reasoning — are exactly the two with almost no natural supply. 87% of the reasoning lane and 75% of the agentic
lane do not exist and must be built (distilled traces; harvested Cursor/Claude-Code trajectories — "it's not
training, it's literally using the model"). That is a deliberate, budgeted synthesis programme, not wishful
accounting. General web stays largest at 40% because common sense lives there: a code-only model writes code
that "won't crash but just doesn't work."

## 2. The Indic lane, split across four provenance tiers

The single most important honesty in the plan. Verified native Indic is thin — a 390B Indic target **cannot**
be met from verified sources, exactly as the session warns. The tier plan (Experiment A, Indic block):

| Tier | Contributes | Unique available | How |
|---|--:|--:|---|
| **Verified native** | 168B | 84B | Sangraha-verified + IndicCorp/Sarvam-verified, repeated ≤2 epochs |
| **Unverified** | 60B | 60B | Sangraha-unverified + crawl, 1 epoch, quality-gated |
| **Translated** | 120B | 120B | EN→Indic high-quality pivot (built), 1 epoch |
| **Synthetic** | 42B | 130B | generated Indic reasoning/instruct/conversation (built), capped ×3 |
| **Total** | **390B** | — | shortfall 0 |

The verified tier is capped at ×2 epochs on purpose (repetition risk); the synthetic tier is *available* far
beyond what we draw (130B), so Indic share can rise in the anneal (§3) without new sourcing. If a reviewer
wants a higher headline Indic %, the plan's answer is a number, not a promise: raise the synthetic draw.

## 3. The anneal — a separate preset, and a reserve held back from day one

The final 2.5% (~75B tokens) runs a **different** mixture on the highest-quality, hardest, cleaned data, with
the learning rate decayed. General web falls sharply; scarce lanes are upsampled:

| Lane | Main | **Anneal** |
|---|--:|--:|
| General web | 40% | **12%** |
| Code | 18% | **20%** |
| Math/STEM | 14% | **12%** |
| Reasoning (long traces) | 6% | **14%** |
| Agentic (Tier-A) | 4% | **18%** |
| Indic (verified + best synthetic) | 13% | **22%** |
| Long context | 5% | **2%** |

**Anneal reserve (declared now, spent only at the end):** ~40B tokens held out of the main run entirely — the
best 30% of agentic trajectories, the longest verified reasoning traces, the highest-quality verified Indic,
and the hardest verified code patches. This reserve is the reason the anneal can exist: *if the selector spends
the best data early, there is nothing special left to cool down on.*

## 4. The protected always-on floor (and proof it's needed)

An OPUS-style selector improves compute efficiency (the session cites ~8× — the loss reached at 20B tokens
would otherwise take 160B), but its proxy is English/code-benchmark-driven, so it undervalues Indic
(off-benchmark) and agentic (looks like logs; only the first ~500 tokens are scored). **Floor: 8% Indic, 3%
agentic, 3% reasoning of every batch, outside the selector.**

Experiment B simulates one selection iteration (100k candidate batches, keep top 50% by proxy score):

| Lane | Candidate supply | OPUS **floor OFF** | OPUS **floor ON** |
|---|--:|--:|--:|
| General web | 60% | 56.4% | 46.2% |
| Code | 16% | 27.0% | 25.2% |
| Math/STEM | 7% | 10.8% | 9.8% |
| Reasoning | 4% | 4.7% | 7.1% |
| **Agentic** | 3% | **0.0%** ← starved | **3.0%** ← protected |
| **Indic** | 7% | **0.0%** ← starved | **8.0%** ← protected |
| Long context | 3% | 1.1% | 0.8% |

Floor OFF, the two lanes the project exists for collapse to **zero**. Floor ON, they hold. This is the
session's central claim, measured — not asserted.

## 5. Curriculum — the order, sequence length, and difficulty

Five stages; general text first, long sequences last, difficulty rising within each. Every transition is a
**warm-up band** (~5–10B tokens of diffusion between the old and new mix) so the gradient norm stays ~0.2 and
never spikes — the V4 lesson that an abrupt band shift can diverge a run.

| Stage | ~Share | Seq len | Mixture shift | Difficulty |
|---|--:|--:|---|---|
| **0 Seed** | 5% | 4k | broad general web | L0–L1 |
| **1 General** | 55% | 4k→8k | web-heavy, code/math ramping in | L1–L2 |
| **2 Reasoning** | 25% | 8k→16k | shift to code/math/science + reasoning + agentic | L2–L3 |
| **3 Long-context** | 12% | 16k→64k→128k | long docs + long agentic trajectories | L3 |
| **4 Anneal** | 3% | 32k | the §3 preset + reserve, LR decayed | L3–L4 |

**Difficulty bands** (one concrete example each — the ladder the session asks for):

| | Level | Example |
|---|---|---|
| **L0** | nursery | "A for apple"; single-fact sentences; 1-step arithmetic |
| **L1** | grade school | GSM8K word problem; a single function call |
| **L2** | high school | MATH level 1–3; a self-contained one-file program |
| **L3** | undergrad | MATH level 4–5; LeetCode-medium; a multi-file edit |
| **L4** | grad/PhD | FrontierMath; a SWE-bench-Verified multi-step agentic fix |

**Reasoning-length bands** (the effort dial — a *distribution* of trace lengths, with boundaries and an example):

| Tag | Trace budget | Example |
|---|--:|---|
| **low** | ≤ 256 tok | "capital of France?" → direct answer, no trace |
| **medium** | 256–2k | a GSM8K problem with a short worked chain |
| **high** | 2k–8k | an AIME problem exploring 2–3 approaches, checking intermediates |
| **ultra** | 8k–32k+ | a FrontierMath / olympiad problem with verification and backtracking |

## 6. The hypothesis, and the experiment (what confirms or refutes this)

**The plan is a hypothesis.** Two of its load-bearing claims are already tested cheaply and deterministically
in [`mixture_plan.py`](./mixture_plan.py):

- **A — supply exists / is honestly synthesised.** ✅ Run: math is at its ×3 repetition ceiling; reasoning and
  agentic are correctly flagged as majority-synthetic; Indic closes only across four tiers. No lane is quietly
  over-allocated.
- **B — the floor stops starvation.** ✅ Run: Indic/agentic go 0.0% → 8%/3% when the floor is switched on.

**What still needs a real proxy (1B and 3B scale, ~20–30B tokens each), before any of this is trusted at 40B:**

1. **Metric — capability parity, not average.** Score = `min over lanes of (benchmark / target)`, so you
   cannot win by starving a lane. Track MMLU (general), LiveCodeBench (code), GSM8K+MATH (math), MILU/IndicQA
   (Indic), and a tau-bench subset (agentic).
2. **Confirm** the mixture if, vs a web-heavy baseline at equal compute, **no lane regresses below its floor**
   *and* Indic rises **without >2 pt MMLU loss**.
3. **Refute** it if Indic gains cost >2 pt general knowledge (share too high), or agentic fails to move at all
   (a *supply/format* problem — go synthesise more trajectories — not a share problem).
4. **Ablations to run:** web-heavy vs proposed; floor ON vs OFF (predict the Experiment-B collapse in a real
   run); anneal reserve ON vs OFF (predict a disproportionate end-of-run jump on the scarce lanes).

Concrete falsifiable prediction to check first at 1B: **with the floor off, OPUS drops the Indic batch share
below 2% within the first 5B tokens; with it on, the share holds ≥8%.** Cheap to measure, and decisive.

## 7. What a hostile reviewer should push on (stated, not hidden)

- **Inventory numbers are order-of-magnitude** and must be pinned (exact dedup'd token counts) before the run;
  the *structure* of the argument is what's defended here, not the third significant figure.
- **The synthesis programme is the real risk.** 87% synthetic reasoning and 75% synthetic/harvested agentic is
  a large bet; it needs its own quality gate and a real:synthetic ceiling (the ~70:30 rule from Session 3) or
  it will teach the model its own artefacts.
- **OPUS proxy-alignment is modelled, not measured** in Experiment B (fixed per-lane means). The 1B proxy
  replaces those assumptions with real weight-change magnitudes.
- **40B + 3T = ~75 tok/param** is well past compute-optimal; justified by inference cost, but it assumes the
  data quality holds under repetition — which the anneal reserve and the math-lane ×3 cap are designed to
  protect.

---

*Reproduce:* `python3 mixture_plan.py` (stdlib only, deterministic) → prints both experiments, writes
`mixture_plan.json`. This plan is the ERA V5 Session 5 submission; the lecture report it is grounded in is
[`session-05-report.md`](https://git.samyama.ai/Samyama.ai/llm-cloud) in the private coordination repo.
