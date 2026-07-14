#!/usr/bin/env python3
"""Parity-aware BPE on the FAITHFUL base — can we beat the reference's fixed-weight 6502?

Same faithful setup as train_faithful.py (NFKC, Metaspace, all chars in vocab -> lossless),
but instead of fixed training weights we select each merge the parity-aware way (Foroutan et al.,
ACL 2026): at every step, take the most-frequent adjacent pair of the language with the CURRENTLY
HIGHEST fertility (tokens / faithful_units). This directly minimizes the spread the score measures.
The 1.2 cap is non-binding (all langs < 0.74), so we run pure parity.

Prints spread/score and the faithfulness gate, and verifies token counts against HF `tokenizers`.
"""
import json, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import regex
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, decoders

HERE = Path(__file__).resolve().parent.parent
LANGS = ["en", "hi", "te", "mai"]
VOCAB = 10000
FU_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
NFKC = normalizers.NFKC()

def metaspace_pieces(text):
    # NFKC then Metaspace(▁, prepend=never): space->▁, each ▁ starts a new piece; \t \n stay literal
    s = NFKC.normalize_str(text).replace(" ", "▁")
    pieces, cur = [], ""
    for c in s:
        if c == "▁":
            if cur: pieces.append(cur)
            cur = "▁"
        else:
            cur += c
    if cur: pieces.append(cur)
    return pieces

RAW  = {l: (HERE / f"corpus/{l}.faithful.txt").read_text(encoding="utf-8") for l in LANGS}
UNITS = {l: len(FU_RE.findall(RAW[l])) for l in LANGS}
PIECES = {l: Counter(metaspace_pieces(RAW[l])) for l in LANGS}

def train_parity():
    reps = {l: {w: list(w) for w in PIECES[l]} for l in LANGS}
    tok_total = {l: sum(len(reps[l][w]) * PIECES[l][w] for w in reps[l]) for l in LANGS}
    paircnt = {l: Counter() for l in LANGS}; pairloc = {l: defaultdict(set) for l in LANGS}
    pairs = lambda s: [(s[i], s[i+1]) for i in range(len(s)-1)]
    for l in LANGS:
        for w, sym in reps[l].items():
            c = PIECES[l][w]
            for p in pairs(sym): paircnt[l][p] += c; pairloc[l][p].add(w)
    vocab = set(ch for l in LANGS for w in PIECES[l] for ch in w) | {"[UNK]"}
    base_chars = len(vocab)
    merges = []
    fert = lambda l: tok_total[l] / UNITS[l]

    def apply_merge(l, a, b):
        ab = a + b
        for w in list(pairloc[l].get((a, b), ())):
            sym = reps[l][w]; c = PIECES[l][w]
            for p in pairs(sym):
                paircnt[l][p] -= c
                if paircnt[l][p] <= 0: paircnt[l].pop(p, None)
                pairloc[l][p].discard(w)
            out, i, removed = [], 0, 0
            while i < len(sym):
                if i < len(sym)-1 and sym[i]==a and sym[i+1]==b: out.append(ab); i+=2; removed+=1
                else: out.append(sym[i]); i+=1
            reps[l][w] = out; tok_total[l] -= removed * c
            for p in pairs(out): paircnt[l][p] += c; pairloc[l][p].add(w)

    while len(vocab) < VOCAB:
        cand = [l for l in LANGS if paircnt[l]]
        if not cand: break
        L = max(cand, key=fert)                      # worst-compressed language
        (a, b), _ = paircnt[L].most_common(1)[0]     # its most valuable pair
        for l in LANGS:
            if (a, b) in pairloc[l]: apply_merge(l, a, b)
        merges.append((a, b)); vocab.add(a + b)
    return vocab, merges, base_chars

vocab, merges, base_chars = train_parity()

# build a real HF tokenizer and MEASURE with it (authoritative)
uv = {"[UNK]": 0}
for t in sorted(vocab - {"[UNK]"}): uv.setdefault(t, len(uv))
tok = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in merges], unk_token="[UNK]"))
tok.normalizer = NFKC
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")

toks = {l: len(tok.encode(RAW[l]).ids) for l in LANGS}
ratios = {l: toks[l]/UNITS[l] for l in LANGS}
spread = max(ratios.values()) - min(ratios.values())
score = 1000/spread

# faithfulness gate (no visible-char loss modulo NFKC)
vis = lambda s: regex.sub(r"\s", "", s)
gate = all(vis(tok.decode(tok.encode(RAW[l]).ids)) == vis(NFKC.normalize_str(RAW[l])) for l in LANGS)

print(f"vocab={tok.get_vocab_size()} (base chars {base_chars})  merges={len(merges)}")
for l in LANGS: print(f"  {l:4s} tokens {toks[l]:>7}  units {UNITS[l]:>7}  fertility {ratios[l]:.6f}")
print(f"spread {spread:.6f}  ->  SCORE {score:.1f}")
print(f"reference/current spread 0.153786 -> 6502.56")
print(f"faithfulness gate: {'PASS' if gate else 'FAIL'}")

if "--save" in sys.argv:
    out = HERE / "experiments/parity_tokenizer.json"
    tok.save(str(out)); print(f"saved {out}")
