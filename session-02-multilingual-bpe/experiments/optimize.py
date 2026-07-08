"""Can we beat 2,336? Sweep BPE-legal levers on top of the H3 script-disjoint structure.

Levers tested (all keep it ONE BPE tokenizer, graders can re-run):
  1. en margin — push English from 1.19 toward the 1.20 cap (raises X1 -> shrinks gap).
  2. finer allocation — denser grid over (Latin budget, en oversample weight, hi, te).
  3. Unicode normalization — NFC / NFD / NFKC baked into the tokenizer's normalizer, applied
     identically at train + eval (and by the grader, since it lives in tokenizer.json). NFD/NFKC
     can change how Brahmic combining marks merge, shifting Hindi/Telugu fertility.

Reports the best feasible score per (normalization, en-margin) so we can see which lever, if any,
moves the needle past the current 2,336.
"""
import os, re, json, unicodedata
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/../data"
LANGS = ["en", "hi", "te", "es"]; VOCAB = 10000
RAW = {l: open(f"{DATA}/{l}.txt", encoding="utf-8").read() for l in LANGS}
WORDS = {l: len(re.findall(r"\S+", RAW[l])) for l in LANGS}          # word count on RAW text
RAWLINES = {l: [ln for ln in RAW[l].split("\n") if ln.strip()] for l in LANGS}

NORMS = {
    "none": None,
    "NFC": normalizers.NFC(),
    "NFD": normalizers.NFD(),
    "NFKC": normalizers.NFKC(),
}
def pynorm(form, s):
    return s if form == "none" else unicodedata.normalize(form, s)

_cache = {}
def train_latin(form, budget, w_en):
    key = ("L", form, budget, w_en)
    if key in _cache: return _cache[key]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    if NORMS[form] is not None: tok.normalizer = NORMS[form]
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=budget, special_tokens=["[UNK]"], show_progress=False)
    def corpus():
        for _ in range(w_en):
            for ln in RAWLINES["en"]: yield ln
        for ln in RAWLINES["es"]: yield ln
    tok.train_from_iterator(corpus(), tr)
    d = json.loads(tok.to_str())["model"]
    _cache[key] = (d["vocab"], [tuple(m) for m in d["merges"]]); return _cache[key]

def train_mono(form, lang, budget):
    key = (lang, form, budget)
    if key in _cache: return _cache[key]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    if NORMS[form] is not None: tok.normalizer = NORMS[form]
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=budget, special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(RAWLINES[lang], tr)
    d = json.loads(tok.to_str())["model"]
    _cache[key] = (d["vocab"], [tuple(m) for m in d["merges"]]); return _cache[key]

def assemble(form, lat_b, w_en, hi_b, te_b):
    uv, seen, mseq = {}, set(), []
    def add(t):
        if t not in uv: uv[t] = len(uv)
    add("[UNK]")
    for vocab, merges in [train_latin(form, lat_b, w_en), train_mono(form, "hi", hi_b), train_mono(form, "te", te_b)]:
        for t in sorted(vocab, key=lambda x: vocab[x]): add(t)
        for ab in merges:
            if ab not in seen: seen.add(ab); mseq.append(ab)
    tok = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in mseq], unk_token="[UNK]"))
    if NORMS[form] is not None: tok.normalizer = NORMS[form]
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return uv, mseq, tok

def fert(tok):
    return {l: len(tok.encode(RAW[l]).tokens) / WORDS[l] for l in LANGS}  # encode RAW; normalizer inside

GRID = [(lb, w, hb, tb)
        for lb in [6100, 6300, 6500]
        for w in [3, 4, 5]
        for hb in [1000, 1200, 1400, 1600]
        for tb in [2200, 2400, 2600, 2800]]

print(f"sweeping {len(GRID)} allocations × {len(NORMS)} norms × 2 margins ...\n")
summary = []
for form in NORMS:
    for margin in [1.19, 1.199]:
        best = None
        for lb, w, hb, tb in GRID:
            uv, mseq, tok = assemble(form, lb, w, hb, tb)
            if len(uv) > VOCAB: continue
            f = fert(tok)
            if f["en"] > margin: continue
            xs = sorted(f.values()); gap = xs[-1] - xs[0]; sc = 1000 / gap
            if best is None or sc > best[0]:
                best = (sc, gap, f, (lb, w, hb, tb), len(uv))
        if best:
            sc, gap, f, alloc, us = best
            summary.append((form, margin, sc, gap, f, alloc, us))
            print(f"[{form:5} en<= {margin}]  score {sc:6.0f}  gap {gap:.3f}  u{us}  "
                  f"en {f['en']:.3f} hi {f['hi']:.3f} te {f['te']:.3f} es {f['es']:.3f}  alloc={alloc}")
        else:
            print(f"[{form:5} en<= {margin}]  no feasible allocation")

print("\n=== best overall (pre-padding; padding adds ~40) ===")
summary.sort(key=lambda r: -r[2])
for form, margin, sc, gap, f, alloc, us in summary[:3]:
    print(f"  {sc:.0f}  norm={form} en<={margin} alloc={alloc}")
print(f"\ncurrent shipped: 2336 (norm=none, en<=1.19, alloc=(6300,4,1200,2600), padded)")
