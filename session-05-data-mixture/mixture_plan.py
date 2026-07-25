#!/usr/bin/env python3
"""ERA V5 · Session 5 — the cheap proxy behind the V5 mixture-and-curriculum plan.

A mixture is a hypothesis until a cheap experiment tests it (the session's refrain).
We can't train a 1B model in a README, but two of the plan's load-bearing claims are
checkable deterministically, right now, from real dataset sizes:

  EXPERIMENT A — SUPPLY.  Does the token supply for each lane actually exist at the
  3.0T budget, or does a share only reach its target by repeating data or synthesising
  it? Sizes real budgets against an approximate dataset inventory and prints the
  repetition factor / synthetic gap per lane, and the Indic tier split.

  EXPERIMENT B — THE ALWAYS-ON FLOOR.  Reproduces the session's central lesson
  numerically: an OPUS-style selector whose proxy is English/code-benchmark-driven
  starves Indic and agentic batches toward zero — UNLESS a protected floor sits
  outside the selector. Simulates one iteration of candidate batches, scores them by
  a per-lane proxy-alignment, keeps the top fraction, and reports the retained lane
  mix with the floor OFF vs ON.

Deterministic (fixed seed). Writes mixture_plan.json for the README to quote.

    python3 mixture_plan.py
"""
import json
import pathlib
import random

random.seed(20260725)
HERE = pathlib.Path(__file__).parent

BUDGET_MAIN_T = 3.0        # trillion tokens, main pretraining (mid of the 2.4-4T brief)
ANNEAL_FRACTION = 0.025    # last 2.5% of the run is the anneal
MODEL = "V5 ~40B India-first (dense or MoE), ~75 tok/param past Chinchilla for inference efficiency"

# ---- Lane targets: share of the MAIN pretraining budget (must sum to 100) ----
MAIN_MIX = {
    "general_web": 40,   # multilingual, English-dominant
    "code":        18,
    "math_stem":   14,
    "reasoning":    6,   # short/medium worked traces (long traces held for anneal)
    "agentic":      4,   # multi-step tool trajectories
    "indic":       13,   # all tiers (see INDIC_TIERS)
    "long_context": 5,   # books + long documents
}
# ---- Anneal mixture: the concentrated final phase (must sum to 100) ----
ANNEAL_MIX = {
    "general_web": 12, "code": 20, "math_stem": 12, "reasoning": 14,
    "agentic": 18, "indic": 22, "long_context": 2,
}
# ---- Protected always-on floor: % of EVERY batch, outside the OPUS selector ----
FLOOR = {"indic": 8, "agentic": 3, "reasoning": 3}

# ---- Approximate dataset inventory (unique tokens, in billions). Order-of-magnitude,
#      to be pinned before the run. Sources are public unless marked. ----
INVENTORY = {
    "general_web": [
        ("FineWeb-Edu (filtered)", 1300), ("DCLM-baseline (hi-Q slice)", 2000),
        ("FineWeb multilingual", 1500),
    ],
    "code": [("The Stack v2 (dedup, permissive)", 900), ("GitHub issues/PRs", 60)],
    "math_stem": [
        ("proof-pile-2 / AlgebraicStack", 55), ("OpenWebMath", 15), ("FineMath", 12),
        ("arXiv + StackExchange STEM", 60),
    ],
    "reasoning": [
        ("OpenThoughts-114k", 1.0), ("OpenR1-Math / NuminaMath-CoT", 3.0),
        ("Bespoke-Stratos / distilled traces", 2.0),
    ],
    "agentic": [
        ("ToolBench", 0.08), ("SWE-Gym / Nebius SWE traces", 1.5),
        ("harvested Cursor/Claude-Code trajectories (built)", 6.0),
    ],
    "long_context": [("books3-style long docs", 40), ("repo-level packed code", 30)],
}
# Indic lane, by provenance tier (unique tokens, billions). Verified native is thin.
INDIC_TIERS = {
    "verified_native": [("Sangraha verified", 64), ("IndicCorp v2 / Sarvam verified", 20)],
    "unverified":      [("Sangraha unverified + crawl", 60)],
    "translated":      [("EN->Indic high-Q pivot (built)", 120)],
    "synthetic":       [("generated Indic reasoning/instruct/convo (built)", 130)],
}

# Benchmarks each lane is meant to win (for the plan; not used in the math)
BENCH = {
    "general_web": "MMLU, MMLU-Pro, GPQA(gen), HellaSwag",
    "code": "SWE-bench Verified/Live/Pro, LiveCodeBench, BFCL",
    "math_stem": "AIME, MATH, GSM8K, FrontierMath",
    "reasoning": "AIME@high, GPQA@high, ARC-AGI",
    "agentic": "SWE-bench(multistep), tau-bench, WebArena, GAIA, BrowseComp, terminal-bench",
    "indic": "MILU, IndicGenBench, IndicQA, FLORES (MT)",
    "long_context": "RULER, LongBench, needle-in-haystack, InfiniteBench",
}

# Max honest repetition before a share is "wishful" (epochs). Web can repeat little
# (abundant); scarce lanes tolerate a few epochs; beyond that => synthesise/gap.
REP_MAX = {"general_web": 1.0, "code": 2.0, "math_stem": 3.0, "reasoning": 4.0,
           "agentic": 4.0, "indic": 3.0, "long_context": 2.0}


def bn(x):  # pretty billions
    return f"{x:,.0f}B" if x >= 1 else f"{x*1000:,.0f}M"


def experiment_a():
    budget_b = BUDGET_MAIN_T * 1000
    rows, verdicts = [], {}
    for lane, share in MAIN_MIX.items():
        target = budget_b * share / 100
        if lane == "indic":
            unique = sum(t for ds in INDIC_TIERS.values() for _, t in ds)
        else:
            unique = sum(t for _, t in INVENTORY[lane])
        rep = target / unique if unique else float("inf")
        rmax = REP_MAX[lane]
        if rep <= 1.0:
            verdict = "unique-covered"
        elif rep <= rmax:
            verdict = f"repeat x{rep:.1f} (<= x{rmax:g} ok)"
        else:
            # portion reachable by repetition to rmax, the rest must be built
            reachable = unique * rmax
            gap = target - reachable
            verdict = f"repeat x{rmax:g} then SYNTHESISE {bn(gap)} ({gap/target*100:.0f}% of lane)"
        rows.append((lane, share, target, unique, rep, verdict))
        verdicts[lane] = {"share_pct": share, "target_B": round(target),
                          "unique_B": round(unique, 1), "rep_factor": round(rep, 2),
                          "verdict": verdict}
    return rows, verdicts


def experiment_a_indic():
    budget_b = BUDGET_MAIN_T * 1000
    indic_target = budget_b * MAIN_MIX["indic"] / 100
    out, total = {}, 0
    for tier, ds in INDIC_TIERS.items():
        u = sum(t for _, t in ds)
        total += u
        out[tier] = u
    # tier plan: use all verified (repeat <=2x), then unverified, translated, synthetic to fill
    plan = {}
    remaining = indic_target
    caps = {"verified_native": 2.0, "unverified": 1.0, "translated": 1.0, "synthetic": 3.0}
    for tier in ["verified_native", "unverified", "translated", "synthetic"]:
        avail = out[tier] * caps[tier]
        take = min(avail, remaining)
        plan[tier] = take
        remaining -= take
    return indic_target, out, plan, remaining


def experiment_b(n=100_000, keep_frac=0.5):
    # supply-proportional candidate pool (English web dominates what a crawl yields)
    supply = {"general_web": 60, "code": 16, "math_stem": 7, "reasoning": 4,
              "agentic": 3, "indic": 7, "long_context": 3}
    lanes = list(supply)
    weights = [supply[l] for l in lanes]
    # OPUS proxy-alignment: an English/code-benchmark proxy values these lanes highly,
    # and undervalues Indic (off-benchmark) and agentic (looks like logs; only first
    # ~500 tokens seen). Mean score per lane on a 0..1 scale.
    align = {"general_web": 0.62, "code": 0.78, "math_stem": 0.74, "reasoning": 0.66,
             "agentic": 0.22, "indic": 0.20, "long_context": 0.5}

    batches = random.choices(lanes, weights=weights, k=n)
    scored = [(l, min(1.0, max(0.0, random.gauss(align[l], 0.15)))) for l in batches]

    # --- floor OFF: pure top-keep_frac by score ---
    kept = sorted(scored, key=lambda x: x[1], reverse=True)[: int(n * keep_frac)]
    off = _mix(kept, lanes)

    # --- floor ON: reserve FLOOR% of the kept budget for protected lanes, fill rest by score ---
    k = int(n * keep_frac)
    on_counts = {l: 0 for l in lanes}
    reserved = 0
    for l, pct in FLOOR.items():
        c = int(k * pct / 100)
        on_counts[l] = c
        reserved += c
    # fill the remaining slots with the top-scored NON-reserved-overflow batches
    pool = sorted(scored, key=lambda x: x[1], reverse=True)
    for l, s in pool:
        if reserved >= k:
            break
        on_counts[l] += 1
        reserved += 1
    on = {l: on_counts[l] / k * 100 for l in lanes}
    return supply, off, on


def _mix(kept, lanes):
    tot = len(kept) or 1
    c = {l: 0 for l in lanes}
    for l, _ in kept:
        c[l] += 1
    return {l: c[l] / tot * 100 for l in lanes}


def main():
    print(f"MODEL: {MODEL}")
    print(f"MAIN BUDGET: {BUDGET_MAIN_T}T tokens  |  ANNEAL: {ANNEAL_FRACTION*100:.1f}% "
          f"= {BUDGET_MAIN_T*1000*ANNEAL_FRACTION:.0f}B tokens\n")

    print("=" * 78)
    print("EXPERIMENT A — SUPPLY (does the budget's data exist?)")
    print("=" * 78)
    rows, verdicts = experiment_a()
    print(f"{'lane':<13}{'share':>6}{'target':>9}{'unique':>9}{'rep':>6}  verdict")
    for lane, share, target, unique, rep, verdict in rows:
        print(f"{lane:<13}{share:>5}%{bn(target):>9}{bn(unique):>9}{rep:>6.1f}  {verdict}")

    itarget, tiers, iplan, irem = experiment_a_indic()
    print(f"\nIndic lane target: {bn(itarget)}  (unique available across tiers: "
          f"{bn(sum(tiers.values()))})")
    for tier, take in iplan.items():
        print(f"  {tier:<16} contribute {bn(take):>8}  (unique {bn(tiers[tier])})")
    print(f"  {'shortfall':<16} {bn(max(0,irem))}  (=> raise synthetic cap or Indic share)")

    print("\n" + "=" * 78)
    print("EXPERIMENT B — ALWAYS-ON FLOOR (does the selector starve Indic/agentic?)")
    print("=" * 78)
    supply, off, on = experiment_b()
    print(f"{'lane':<13}{'supply':>8}{'OPUS off':>10}{'OPUS on':>9}  effect")
    for l in supply:
        d = on[l] - off[l]
        note = ""
        if l in FLOOR and off[l] < 2.0:
            note = f"STARVED -> protected to {on[l]:.0f}%"
        print(f"{l:<13}{supply[l]:>7}%{off[l]:>9.1f}%{on[l]:>8.1f}%  {note}")

    out = {
        "model": MODEL, "main_budget_T": BUDGET_MAIN_T,
        "anneal_fraction": ANNEAL_FRACTION,
        "main_mix_pct": MAIN_MIX, "anneal_mix_pct": ANNEAL_MIX,
        "always_on_floor_pct": FLOOR, "benchmarks": BENCH,
        "experiment_a_supply": verdicts,
        "indic_tier_plan_B": {k: round(v) for k, v in iplan.items()},
        "indic_target_B": round(itarget),
        "experiment_b_floor": {
            "candidate_supply_pct": supply,
            "retained_floor_off_pct": {k: round(v, 1) for k, v in off.items()},
            "retained_floor_on_pct": {k: round(v, 1) for k, v in on.items()},
        },
    }
    (HERE / "mixture_plan.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'mixture_plan.json'}")


if __name__ == "__main__":
    main()
