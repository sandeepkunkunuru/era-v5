"""VoCap-style allocation (Zheng et al. 2021, EMNLP) adapted to the fertility-parity objective.

VoCap sets per-language vocabulary capacity by marginal utility instead of fixed heuristics. Our
objective is min-max fertility (not coverage/ALP), so the faithful analog is greedy water-filling:
start each script group at its base, then repeatedly add a budget chunk to whichever group most
reduces the current MAX fertility of {hi, te, es}, subject to en <= 1.19. Then apply the H4 whole-word
reclamation on top. Compares the principled allocation to the hand-grid winner (base 6300/1100/2200 ->
score 2,430).
"""
import os, re, json
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/../data"
LANGS = ["en", "hi", "te", "es"]; VOCAB = 10000; EN_MARGIN = 1.19; W_EN = 4
RAW = {l: open(f"{DATA}/{l}.txt", encoding="utf-8").read() for l in LANGS}
WORDS = {l: len(re.findall(r"\S+", RAW[l])) for l in LANGS}
FREQ = {l: Counter(re.findall(r"\S+", RAW[l])) for l in LANGS}
LINES = {l: [ln for ln in RAW[l].split("\n") if ln.strip()] for l in LANGS}

_c = {}
def train_latin(b, w):
    k = ("L", b, w)
    if k in _c: return _c[k]
    t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=b, special_tokens=["[UNK]"], show_progress=False)
    def corp():
        for _ in range(w):
            for ln in LINES["en"]: yield ln
        for ln in LINES["es"]: yield ln
    t.train_from_iterator(corp(), tr); d = json.loads(t.to_str())["model"]
    _c[k] = (d["vocab"], [tuple(m) for m in d["merges"]]); return _c[k]
def train_mono(l, b):
    k = (l, b)
    if k in _c: return _c[k]
    t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=b, special_tokens=["[UNK]"], show_progress=False)
    t.train_from_iterator(LINES[l], tr); d = json.loads(t.to_str())["model"]
    _c[k] = (d["vocab"], [tuple(m) for m in d["merges"]]); return _c[k]

def union(lat_b, hi_b, te_b):
    uv, seen, mseq = {}, set(), []
    def add(x):
        if x not in uv: uv[x] = len(uv)
    add("[UNK]")
    for vocab, merges in [train_latin(lat_b, W_EN), train_mono("hi", hi_b), train_mono("te", te_b)]:
        for t in sorted(vocab, key=lambda x: vocab[x]): add(t)
        for ab in merges:
            if ab not in seen: seen.add(ab); mseq.append(ab)
    return uv, mseq
def mk(uv, mseq):
    t = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in mseq], unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit(); return t
def fert(tok): return {l: len(tok.encode(RAW[l]).tokens) / WORDS[l] for l in LANGS}

def reclaim(uv, mseq, targets=("es", "hi", "te")):
    tok = mk(uv, mseq)
    order = {l: [w for w, _ in FREQ[l].most_common()] for l in targets}; targets = list(targets); r = 0
    while len(uv) < VOCAB and r < 80:
        r += 1; f = fert(tok); lang = max(targets, key=lambda l: f[l]); added = 0; lim = min(120, VOCAB - len(uv))
        for w in order[lang]:
            if added >= lim or len(uv) >= VOCAB: break
            enc = tok.encode(w).tokens
            if len(enc) < 2: continue
            a, b = enc[0], enc[1]; p = a + b
            if p in uv: continue
            uv[p] = len(uv); mseq.append((a, b)); added += 1
        if added == 0:
            targets = [l for l in targets if l != lang]
            if not targets: break
            continue
        tok = mk(uv, mseq)
    return tok

def score(f): xs = sorted(f.values()); g = xs[-1] - xs[0]; return g, (1000 / g if g > 0 else 9e9)

# ---- VoCap-style greedy water-filling for the BASE allocation ----
CHUNK = 150; BASE_TARGET = 9300           # leave ~700 slots for reclamation
b = {"lat": 5800, "hi": 400, "te": 700}   # start near the en-cap knee; hi/te at ~base coverage
def base_size(bb):
    uv, _ = union(bb["lat"], bb["hi"], bb["te"]); return len(uv)
step = 0
while sum(b.values()) < BASE_TARGET and step < 60:
    step += 1
    cur = union(b["lat"], b["hi"], b["te"]); f = fert(mk(*cur))
    if f["en"] > EN_MARGIN:                # need more Latin to satisfy the cap first
        b["lat"] += CHUNK; continue
    best_g, best_max = None, 9e9
    for g in ["lat", "hi", "te"]:
        b2 = dict(b); b2[g] += CHUNK
        uv2, ms2 = union(b2["lat"], b2["hi"], b2["te"])
        if len(uv2) > VOCAB: continue
        f2 = fert(mk(uv2, ms2))
        if f2["en"] > EN_MARGIN: continue
        m = max(f2["hi"], f2["te"], f2["es"])   # min-max objective on the non-capped languages
        if m < best_max: best_max, best_g = m, g
    if best_g is None: break
    b[best_g] += CHUNK

uv, mseq = union(b["lat"], b["hi"], b["te"])
fb = fert(mk(uv, mseq)); gb, sb = score(fb)
print(f"VoCap-style BASE alloc: Lat={b['lat']} (x{W_EN}) hi={b['hi']} te={b['te']}  union={len(uv)}")
print(f"  base fert: en {fb['en']:.3f} hi {fb['hi']:.3f} te {fb['te']:.3f} es {fb['es']:.3f}  gap {gb:.3f} score {sb:.0f}")
tok = reclaim(uv, mseq)
fr = fert(tok); gr, sr = score(fr)
print(f"  + reclamation -> u{len(tok.get_vocab())}: en {fr['en']:.3f} hi {fr['hi']:.3f} te {fr['te']:.3f} es {fr['es']:.3f}  gap {gr:.3f}  SCORE {sr:.0f}")
print(f"\n  hand-grid H4 baseline: score 2430 (Lat 6300 hi1100 te2200)")
print(f"  VoCap-style result:    score {sr:.0f}  -> {'BEATS 2430' if sr > 2430 else 'no improvement over grid'}")
