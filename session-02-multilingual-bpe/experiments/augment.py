"""Beat 2,336 by reallocating wasted BPE budget to whole words (Picky-BPE / BPE-knockout idea).

Diagnosis: the shipped 10k tokenizer leaves 17% of slots on intermediate merges that never
surface as final tokens, while whole-word coverage is low (hi 21%, es 17%). So: train the
script-disjoint union at a REDUCED budget (fewer intermediates), then spend the freed slots on
explicit whole-word merges for the highest-fertility languages — greedily collapsing their most
frequent multi-token words to single tokens. Still ONE valid BPE tokenizer (vocab + ordered
merges), so graders can re-run it.
"""
import os, re, json
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/../data"
LANGS = ["en", "hi", "te", "es"]; VOCAB = 10000
RAW = {l: open(f"{DATA}/{l}.txt", encoding="utf-8").read() for l in LANGS}
WORDS = {l: len(re.findall(r"\S+", RAW[l])) for l in LANGS}
FREQ = {l: Counter(re.findall(r"\S+", RAW[l])) for l in LANGS}
LINES = {l: [ln for ln in RAW[l].split("\n") if ln.strip()] for l in LANGS}

_c = {}
def train_latin(b, w_en):
    k = ("L", b, w_en)
    if k in _c: return _c[k]
    t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=b, special_tokens=["[UNK]"], show_progress=False)
    def corp():
        for _ in range(w_en):
            for ln in LINES["en"]: yield ln
        for ln in LINES["es"]: yield ln
    t.train_from_iterator(corp(), tr); d = json.loads(t.to_str())["model"]
    _c[k] = (d["vocab"], [tuple(m) for m in d["merges"]]); return _c[k]
def train_mono(lang, b):
    k = (lang, b)
    if k in _c: return _c[k]
    t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=b, special_tokens=["[UNK]"], show_progress=False)
    t.train_from_iterator(LINES[lang], tr); d = json.loads(t.to_str())["model"]
    _c[k] = (d["vocab"], [tuple(m) for m in d["merges"]]); return _c[k]

def base_union(lat_b, w_en, hi_b, te_b):
    uv, seen, mseq = {}, set(), []
    def add(t):
        if t not in uv: uv[t] = len(uv)
    add("[UNK]")
    for vocab, merges in [train_latin(lat_b, w_en), train_mono("hi", hi_b), train_mono("te", te_b)]:
        for t in sorted(vocab, key=lambda x: vocab[x]): add(t)
        for ab in merges:
            if ab not in seen: seen.add(ab); mseq.append(ab)
    return uv, mseq

def make(uv, mseq):
    t = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in mseq], unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit(); return t

def fert(tok): return {l: len(tok.encode(RAW[l]).tokens) / WORDS[l] for l in LANGS}
def score(f): xs = sorted(f.values()); g = xs[-1] - xs[0]; return g, (1000/g if g > 0 else 9e9)

def greedy_augment(lat_b, w_en, hi_b, te_b, targets, en_cap=1.2):
    """Fill slots up to VOCAB with whole-word merges for the highest-fertility target languages."""
    uv, mseq = base_union(lat_b, w_en, hi_b, te_b)
    if len(uv) > VOCAB: return None
    tok = make(uv, mseq)
    # precompute, per target lang, its words sorted by frequency (desc)
    order = {l: [w for w, _ in FREQ[l].most_common()] for l in targets}
    rounds = 0
    while len(uv) < VOCAB and rounds < 60:
        rounds += 1
        f = fert(tok)
        if f["en"] > en_cap: return None
        lang = max(targets, key=lambda l: f[l])           # attack the current worst language
        added = 0
        budget_this_round = min(120, VOCAB - len(uv))     # add in chunks, then rebuild
        for w in order[lang]:
            if added >= budget_this_round or len(uv) >= VOCAB: break
            enc = tok.encode(w).tokens
            if len(enc) < 2: continue
            a, b = enc[0], enc[1]; prod = a + b            # collapse the first adjacent pair
            if prod in uv: continue
            uv[prod] = len(uv); mseq.append((a, b)); added += 1
        if added == 0:                                     # this lang fully collapsed; drop it
            targets = [l for l in targets if l != lang]
            if not targets: break
            continue
        tok = make(uv, mseq)
    return tok, fert(tok), len(uv)

print("baseline shipped: score 2336 (gap 0.428)\n")
configs = [
    # (lat_b, w_en, hi_b, te_b, targets)
    (6300, 4, 700, 1600, ["es", "hi", "te"]),
    (6300, 4, 900, 1800, ["es", "hi", "te"]),
    (6300, 4, 1100, 2200, ["es", "hi", "te"]),
    (6500, 4, 700, 1600, ["es", "hi", "te"]),
    (6100, 4, 900, 1800, ["es", "hi", "te"]),
    (6300, 4, 500, 1200, ["es", "hi", "te"]),
]
best = None
for lat_b, w_en, hi_b, te_b, targets in configs:
    r = greedy_augment(lat_b, w_en, hi_b, te_b, targets)
    if not r:
        print(f"Lat({lat_b},x{w_en}) hi{hi_b} te{te_b}: infeasible (en cap or over-budget)"); continue
    tok, f, us = r
    g, sc = score(f)
    tag = "  <== beats 2336" if sc > 2336 else ""
    print(f"Lat({lat_b},x{w_en}) hi{hi_b} te{te_b} -> u{us}  "
          f"en {f['en']:.3f} hi {f['hi']:.3f} te {f['te']:.3f} es {f['es']:.3f}  gap {g:.3f} score {sc:.0f}{tag}")
    if best is None or sc > best[0]: best = (sc, g, f, (lat_b, w_en, hi_b, te_b), tok)

print(f"\nBEST augmented score: {best[0]:.0f}  (gap {best[1]:.3f})  alloc={best[3]}")
print(f"  vs shipped 2336 -> {'IMPROVEMENT' if best[0] > 2336 else 'no improvement'}")
