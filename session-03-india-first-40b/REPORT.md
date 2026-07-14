# A 40B India-First Model — Design

**Target:** a 40B model, Gemma-3-class, excellent at **coding, agentic work, and Indic languages**, that
reasons from an **Indian** vantage. Below are the decisions; every line is a choice **and** its reason.
India-first is treated as three things, not translation: **corpus composition + a values constitution +
tokenizer efficiency**.

## Decisions (TL;DR)

| Decision | Value | One-line reason |
|---|---|---|
| Languages | English + 12 Indic: hi bn mr te ta gu kn ml pa or ur as | ≈ the scheduled-language speaker mass; grouped by script for shared budget |
| **Vocab size** | **256k** (floor 200k) | measured fertility knee ≈200k + Tao law (40B → ~200k) + Aya/Gemma-3 precedent (256k/262k); round up for breadth |
| Pre-train mix | En 40 / Indic 25 / Code 18 / Math+Sci 10 / Other 7 (%) | over-weight Indic (~25% vs ~1% web share) to force transfer; code+math (28%) drive coding/agentic |
| Tokenizer | SentencePiece-style BPE, NFKC, byte-fallback, digit-split | Indic-atomic (not byte-level) → low Indic fertility, lossless on all scripts |
| Post-train | SFT → DPO → RLVR (Tülu 3) + Constitutional-AI Indian-values | open SOTA recipe; verifiable reward for code/math; values ≠ translation |

## 1. Fertility → vocab size (measured, not asserted)

We trained candidate BPE tokenizers (NFKC + Metaspace, Indic-atomic) at 32k–256k on an Indic-heavy
mix (en+12 Indic Wikipedia + code + math) and measured fertility on **FLORES-200 devtest** — parallel
text, so differences are script/tokenizer, not content. Fertility = tokens/word; **premium** = ×English.
Harness validates against Sarvam-1 (our 64k: indic-mean 2.06 / en 1.41, inside their reported 1.4–2.1).

**Fertility vs vocab (indic mean → knee):** 2.40 (32k) → 2.06 (64k) → 1.81 (128k) → **1.68 (200k)** →
1.63 (256k). The last 56k of vocab buys only −0.06 — the knee is ~200k.

**Per-language fertility @ 256k** (grouped by script; premium ×English in parens):

| Perso-Arabic | Devanagari | Bengali script | Gurmukhi/Odia | Dravidian (Brahmic) |
|---|---|---|---|---|
| ur 1.23 (1.02×) | **hi 1.31 (1.08×)**, mr 1.60 (1.32×) | bn 1.49 (1.23×), as 1.56 (1.29×) | pa 1.32 (1.09×), or 1.63 (1.35×) | gu 1.50, te 1.84, ta 1.84, kn 1.98, **ml 2.19 (1.81×)** |

en = 1.21; code 377 tok/1k-char, math 233 tok/1k-char.

**Targets & derivation.** Target Indic premium **≤ 1.5×** English. Reachable for Indo-Aryan/Perso-Arabic
(hi, ur, pa, bn) but **not for Dravidian** (ml/kn/te/ta stay 1.8–2.2× at every vocab): *morphology, not
script, is the binding constraint* — agglutinative languages have long words, so subword vocab alone can't
equalize them; that needs morpheme-aware pre-tokenization or a per-script budget boost (noted as a
limitation, not hidden). The vocab number: the **measured knee (~200k)** and **Tao-optimal for 40B (~200k)**
agree; breadth (13 langs, 10 scripts, stubborn Dravidian tail) and negligible embedding cost (256k × ~6k
hidden ≈ 1.6B params, ~4% of 40B, tied) push us to round up to **256k** — matching Gemma-3/Aya, under which
the last increment still trims the worst-language premium (1.85×→1.81×).

## 2. Data

**Pre-training.** Mix above; set weights with **DoReMi/DSIR** (learned domain weights), not by hand.
Within the Indic 25%, temperature-smooth (α≈0.3) by *verified* token count so Hindi/Bengali don't crowd
out the tail — but **cap synthetic** share: Sangraha is 251B total yet verified Assamese is only ~0.3B, so
a synthetic-heavy tail teaches translationese. English anchors reasoning; code = The Stack v2 (license +
near-dedup); math = DeepSeekMath-style recall + OpenMathInstruct-2.

**Post-training.** **SFT** (Tülu-3 mixture + Indic instructions, Airavata-style) → **DPO** (preference,
reward-model-free) → **RLVR** (verifier reward: unit tests for code, answer-check for math). Agentic data =
**ReAct** traces + tool-use trajectories (ToolLLM/Gorilla), multi-turn.

**RL / alignment.** **Constitutional AI** with an explicit **Indian-values constitution** (pluralism,
Indian legal/social norms, code-mixing tolerance) → RLAIF — this, not translation, is what makes the model
"India-first" in worldview. RLVR for verifiable skills; DPO for helpfulness/safety.

## 3. Cleaning (pipeline, per objective)

`lang-ID (fastText/CLD3)` → `CCNet dedup + perplexity segmentation` → `Gopher quality heuristics` →
`MinHash near-dedup + exact-substring dedup` (Lee 2022: ~10× less memorization) → `benchmark
decontamination (n-gram)` → `PII + toxicity scrub`. **Indic-specific:** Unicode NFC + script/transliteration
normalization (indic-nlp-library), and a FineWeb-Edu-style **quality classifier trained on Indic** (the
edu-classifier is English-only today). **Code:** license filter + near-dedup + optional execution filter.

## 4. Evaluation (per objective, decontaminated)

| Objective | Benchmarks |
|---|---|
| Indic | MILU, IndicGenBench, IndicXTREME, Airavata-eval |
| Code / agentic | HumanEval, MBPP, MultiPL-E, LiveCodeBench, **SWE-bench** |
| Math / science | GSM8K, MATH, GPQA |
| General | MMLU + Indic-MMLU |
| Agentic | AgentBench, τ-bench |

**Decontamination discipline:** strip any train doc with ≥13-gram / 50-char overlap vs every eval set
before training; prefer **time-split live** benchmarks (LiveCodeBench) as leakage insurance.

## 5. Why this is "India-first"

Not translation. Three levers together: (1) **composition** — Indic over-weighted ~25× its web share;
(2) **worldview** — an Indian-values constitution shapes RLAIF; (3) **efficiency** — an Indic-atomic
tokenizer so Indian users don't pay a 3–5× token tax (Petrov 2023). The vocab number is the hinge: it is
what converts "supports Indic" into "cheap in Indic."

---

_Sources: Gemma 3 (2503.19786), Tao et al. Scaling-Laws-with-Vocabulary (2407.13623, NeurIPS 2024), Petrov
et al. tokenizer unfairness (2305.15425, NeurIPS 2023), Sarvam-1, Sangraha/IndicCorp v2 (2212.05409),
FineWeb (2406.17557), Dolma (2402.00159), Tülu 3 (2411.15124), DoReMi (2305.10429), Constitutional AI
(2212.08073), The Stack v2 (2402.19173), DeepSeekMath (2402.03300). Fertility numbers measured on
FLORES-200; harness in this repo._
