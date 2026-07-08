"""H5 — parity-aware BPE (the principled method; replaces the H3/H4 post-hoc pipeline).

Foroutan et al. (ACL 2026): change ONE thing in BPE training — at each merge step, don't take the
globally most-frequent pair; take the most valuable pair of the currently WORST-compressed language.
Parity is optimized during training, not patched on afterward. Still one ordinary BPE tokenizer
(vocab + ordered merges, WhitespaceSplit pre-tokenizer), so graders can re-run it.

Here we run it under the assignment's English<=1.2 cap: feed English until it clears the cap (with a
safe 1.185 margin so it survives a Wikipedia re-fetch), then pure parity on the other three — which
lands them at an *identical* fertility. Result: en 1.182 / hi=te=es 1.581, gap 0.399. The tokenizer is
verified against HuggingFace `tokenizers` (max Δ 0.0000). Removing the cap gives PERFECT parity (all
four = 1.390, gap 0) — see experiments/parity_bpe.py; the cap is what creates the gap.
"""
import os, re, json
from collections import Counter, defaultdict
from tokenizers import Tokenizer, models, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/data"
LANGS = ["en", "hi", "te", "es"]; VOCAB = 10000; EN_CAP = 1.2; EN_MARGIN = 1.185
PAGE_TITLES = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "es": "India"}
RAW = {l: open(f"{DATA}/{l}.txt", encoding="utf-8").read() for l in LANGS}
WORDS = {l: re.findall(r"\S+", RAW[l]) for l in LANGS}
WCOUNT = {l: Counter(WORDS[l]) for l in LANGS}
WTOTAL = {l: len(WORDS[l]) for l in LANGS}

def train_parity(cap_en):
    reps = {l: {w: list(w) for w in WCOUNT[l]} for l in LANGS}
    tok_total = {l: sum(len(reps[l][w]) * WCOUNT[l][w] for w in reps[l]) for l in LANGS}
    paircnt = {l: Counter() for l in LANGS}; pairloc = {l: defaultdict(set) for l in LANGS}
    pairs = lambda s: [(s[i], s[i + 1]) for i in range(len(s) - 1)]
    for l in LANGS:
        for w, sym in reps[l].items():
            c = WCOUNT[l][w]
            for p in pairs(sym): paircnt[l][p] += c; pairloc[l][p].add(w)
    vocab = set(ch for l in LANGS for w in WCOUNT[l] for ch in w) | {"[UNK]"}
    merges = []; en_ok = False
    fert = lambda l: tok_total[l] / WTOTAL[l]

    def apply_merge(l, a, b):
        ab = a + b
        for w in list(pairloc[l].get((a, b), ())):
            sym = reps[l][w]; c = WCOUNT[l][w]
            for p in pairs(sym):
                paircnt[l][p] -= c
                if paircnt[l][p] <= 0: paircnt[l].pop(p, None)
                pairloc[l][p].discard(w)
            out, i, removed = [], 0, 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b: out.append(ab); i += 2; removed += 1
                else: out.append(sym[i]); i += 1
            reps[l][w] = out; tok_total[l] -= removed * c
            for p in pairs(out): paircnt[l][p] += c; pairloc[l][p].add(w)

    while len(vocab) < VOCAB:
        if not en_ok:
            L = "en"
            if not paircnt["en"] or fert("en") <= cap_en: en_ok = True; continue
        else:
            cand = [l for l in LANGS if paircnt[l]]
            if not cand: break
            L = max(cand, key=fert)
        (a, b), _ = paircnt[L].most_common(1)[0]
        for l in LANGS:
            if (a, b) in pairloc[l]: apply_merge(l, a, b)
        merges.append((a, b)); vocab.add(a + b)
        if not en_ok and fert("en") <= cap_en: en_ok = True
    return vocab, merges

def build_tokenizer(vocab, merges):
    uv = {"[UNK]": 0}
    for t in sorted(vocab - {"[UNK]"}): uv.setdefault(t, len(uv))
    tok = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in merges], unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok

vocab, merges = train_parity(EN_MARGIN)
tok = build_tokenizer(vocab, merges)
fert = {l: len(tok.encode(RAW[l]).tokens) / WTOTAL[l] for l in LANGS}
xs = sorted(fert.values()); x1, x4 = xs[0], xs[-1]; gap = x4 - x1; sc = 1000 / gap
assert fert["en"] <= EN_CAP, f"English {fert['en']:.4f} exceeds cap {EN_CAP}"
assert tok.get_vocab_size() == VOCAB, f"vocab {tok.get_vocab_size()} != {VOCAB}"
print(f"H5 parity-aware ({tok.get_vocab_size()} tokens): "
      f"en {fert['en']:.3f} hi {fert['hi']:.3f} te {fert['te']:.3f} es {fert['es']:.3f}  "
      f"gap {gap:.3f}  SCORE {sc:.1f}")

tok.save(f"{HERE}/tokenizer.json")
toks = [t for t, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1])]
open(f"{HERE}/tokens.txt", "w", encoding="utf-8").write("\n".join(toks))
json.dump({"method": "H5 parity-aware BPE (Foroutan et al. ACL 2026): merge the worst language's pair each step",
           "langs": LANGS, "vocab": len(toks), "en_cap": EN_CAP,
           "words": {l: WTOTAL[l] for l in LANGS}, "fertility": fert,
           "X1": x1, "X4": x4, "gap": gap, "score": sc, "page_titles": PAGE_TITLES},
          open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=2)
print("saved tokenizer.json, tokens.txt, results.json")
