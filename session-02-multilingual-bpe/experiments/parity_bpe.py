"""Parity-aware BPE from first principles (Foroutan et al., ACL 2026 — the principled version of H3/H4).

The problem, stated cleanly: build ONE subword vocabulary so that fertility (tokens-per-word) is as
EQUAL as possible across languages — no language pays a token tax. Standard BPE picks, at each step,
the globally most-frequent adjacent pair → it serves whichever language dominates the corpus.

Parity-aware BPE changes ONE thing: at each merge step, look at the currently WORST-compressed
(highest-fertility) language and merge *its* most-valuable pair. Parity is optimized during training,
not patched on afterward. The result is still one ordinary BPE tokenizer (vocab + ordered merges).

This is a self-contained, correct trainer (incremental pair counts). We run it two ways:
  (A) pure parity — minimize the spread, no English cap → the genuinely FAIR tokenizer.
  (B) with the assignment's English<=1.2 cap → feed English first until satisfied, then pure parity,
      to show what the cap costs in fairness.
The saved tokenizer is verified against HuggingFace `tokenizers` so the numbers are honest.
"""
import os, re, json
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/../data"
LANGS = ["en", "hi", "te", "es"]; VOCAB = 10000
RAW = {l: open(f"{DATA}/{l}.txt", encoding="utf-8").read() for l in LANGS}
WORDS = {l: re.findall(r"\S+", RAW[l]) for l in LANGS}
WCOUNT = {l: Counter(WORDS[l]) for l in LANGS}          # word type -> freq on the page
WTOTAL = {l: len(WORDS[l]) for l in LANGS}

def train_parity(cap_en=None, verbose=False):
    # per-language state: word -> list of symbols; incremental pair counts + locations
    reps = {l: {w: list(w) for w in WCOUNT[l]} for l in LANGS}
    tok_total = {l: sum(len(reps[l][w]) * WCOUNT[l][w] for w in reps[l]) for l in LANGS}
    paircnt = {l: Counter() for l in LANGS}
    pairloc = {l: defaultdict(set) for l in LANGS}
    def word_pairs(sym):
        return [(sym[i], sym[i + 1]) for i in range(len(sym) - 1)]
    for l in LANGS:
        for w, sym in reps[l].items():
            c = WCOUNT[l][w]
            for p in word_pairs(sym):
                paircnt[l][p] += c; pairloc[l][p].add(w)

    vocab = set(ch for l in LANGS for w in WCOUNT[l] for ch in w) | {"[UNK]"}
    merges = []
    en_ok = cap_en is None

    def fert(l): return tok_total[l] / WTOTAL[l]

    def apply_merge(l, a, b):
        ab = a + b; affected = list(pairloc[l].get((a, b), ()))
        for w in affected:
            sym = reps[l][w]; c = WCOUNT[l][w]
            for p in word_pairs(sym):                    # retract old pairs
                paircnt[l][p] -= c
                if paircnt[l][p] <= 0: paircnt[l].pop(p, None)
                pairloc[l][p].discard(w)
            out, i, removed = [], 0, 0                    # do the merge (non-overlapping, L->R)
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(ab); i += 2; removed += 1
                else:
                    out.append(sym[i]); i += 1
            reps[l][w] = out; tok_total[l] -= removed * c
            for p in word_pairs(out):                     # add new pairs
                paircnt[l][p] += c; pairloc[l][p].add(w)

    while len(vocab) < VOCAB:
        # choose the target language
        if not en_ok:
            L = "en"                                       # satisfy the cap first
        else:
            cand = [l for l in LANGS if paircnt[l]]
            if not cand: break
            L = max(cand, key=fert)                        # the worst-compressed language
        if not paircnt[L]:
            if L == "en": en_ok = True; continue
            break
        (a, b), _ = paircnt[L].most_common(1)[0]           # its most valuable merge
        ab = a + b
        # apply to EVERY language that has this pair (en/es share Latin)
        for l in LANGS:
            if (a, b) in pairloc[l]: apply_merge(l, a, b)
        merges.append((a, b)); vocab.add(ab)
        if not en_ok and fert("en") <= cap_en: en_ok = True
        if verbose and len(vocab) % 1500 == 0:
            print(f"  |V|={len(vocab)} " + " ".join(f"{l}={fert(l):.3f}" for l in LANGS))

    fert_final = {l: fert(l) for l in LANGS}
    return vocab, merges, fert_final

def verify_with_hf(vocab, merges):
    from tokenizers import Tokenizer, models, pre_tokenizers
    uv = {"[UNK]": 0}
    for t in sorted(vocab - {"[UNK]"}): uv.setdefault(t, len(uv))
    # ensure every merge product present (it is) and give a stable id ordering
    tok = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in merges], unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return {l: len(tok.encode(RAW[l]).tokens) / WTOTAL[l] for l in LANGS}, tok

def report(name, fert, hf=None):
    xs = sorted(fert.values()); gap = xs[-1] - xs[0]
    line = " ".join(f"{l}={fert[l]:.3f}" for l in LANGS)
    print(f"{name}: {line}  gap={gap:.3f}  score={1000/gap:.0f}")
    if hf: print(f"    HF re-check: " + " ".join(f"{l}={hf[l]:.3f}" for l in LANGS) +
                 f"  (max Δ {max(abs(hf[l]-fert[l]) for l in LANGS):.4f})")

if __name__ == "__main__":
    print("(A) PURE parity-aware BPE — no English cap (the genuinely fair tokenizer)")
    vA, mA, fA = train_parity(cap_en=None, verbose=True)
    hfA, tokA = verify_with_hf(vA, mA)
    report("  pure-parity", fA, hfA)

    print("\n(B) parity-aware BPE with English<=1.2 cap (assignment regime)")
    vB, mB, fB = train_parity(cap_en=1.2, verbose=True)
    hfB, tokB = verify_with_hf(vB, mB)
    report("  capped-parity", fB, hfB)

    print("\nfor reference: H4 post-hoc = en1.180 hi1.582 te1.591 es1.577 gap0.411 score2430")
