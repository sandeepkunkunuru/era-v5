"""Task 1 — does parity-aware BPE's perfect parity survive at scale / off the India pages?

The 4-language result (all converge to one fertility) could be an artifact of a small, curated,
mostly-disjoint-script set. Here we stress it: fetch the *same* article ("India") in ~14 languages
spanning Latin, Cyrillic, Brahmic, Arabic, and CJK, and run PURE parity-aware BPE (Foroutan ACL 2026
style — each merge helps the worst-compressed language) at a fixed joint vocabulary.

Questions:
  1. Does the fertility SPREAD stay ~0 as the number of languages and script diversity grows?
  2. How does the common fertility move with N languages at fixed budget?
  3. Does the whole "tokens per \\S+ word" framework even apply to CJK/Thai (no whitespace)?

Corpora are cached in data_scale/ for reproducibility.
"""
import os, re, json, urllib.request, urllib.parse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = f"{HERE}/../data_scale"
os.makedirs(CACHE, exist_ok=True)

# (lang, wiki-subdomain, article title) — the "India" article across scripts
LANGSET = [
    ("en", "en", "India"), ("es", "es", "India"), ("fr", "fr", "Inde"), ("de", "de", "Indien"),
    ("pt", "pt", "Índia"), ("id", "id", "India"),                    # Latin
    ("ru", "ru", "Индия"),                                            # Cyrillic
    ("ar", "ar", "الهند"),                                            # Arabic (RTL, whitespace)
    ("hi", "hi", "भारत"), ("te", "te", "భారతదేశం"), ("ta", "ta", "இந்தியா"), ("bn", "bn", "ভারত"),  # Brahmic
    ("zh", "zh", "印度"), ("ja", "ja", "インド"),                      # CJK — no whitespace
]

def fetch(sub, title):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                "redirects": 1, "format": "json", "titles": title})
    req = urllib.request.Request(f"https://{sub}.wikipedia.org/w/api.php?{q}",
                                 headers={"User-Agent": "era-v5-research/1.0 (educational)"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return next(iter(d["query"]["pages"].values())).get("extract", "")

def load():
    texts = {}
    for lang, sub, title in LANGSET:
        p = f"{CACHE}/{lang}.txt"
        if not os.path.exists(p):
            try: open(p, "w", encoding="utf-8").write(fetch(sub, title))
            except Exception as e: print(f"  fetch {lang} failed: {e}"); continue
        texts[lang] = open(p, encoding="utf-8").read()
    return texts

# ---- generalized PURE parity-aware BPE over an arbitrary set of languages ----
def parity_bpe(texts, V):
    langs = list(texts)
    WORDS = {l: re.findall(r"\S+", texts[l]) for l in langs}
    WCOUNT = {l: Counter(WORDS[l]) for l in langs}
    WTOTAL = {l: max(1, len(WORDS[l])) for l in langs}
    reps = {l: {w: list(w) for w in WCOUNT[l]} for l in langs}
    tok_total = {l: sum(len(reps[l][w]) * WCOUNT[l][w] for w in reps[l]) for l in langs}
    paircnt = {l: Counter() for l in langs}; pairloc = {l: defaultdict(set) for l in langs}
    pairs = lambda s: [(s[i], s[i + 1]) for i in range(len(s) - 1)]
    for l in langs:
        for w, sym in reps[l].items():
            c = WCOUNT[l][w]
            for p in pairs(sym): paircnt[l][p] += c; pairloc[l][p].add(w)
    vocab = set(ch for l in langs for w in WCOUNT[l] for ch in w) | {"[UNK]"}
    fert = lambda l: tok_total[l] / WTOTAL[l]
    def apply_merge(l, a, b):
        ab = a + b
        for w in list(pairloc[l].get((a, b), ())):
            sym = reps[l][w]; c = WCOUNT[l][w]
            for p in pairs(sym):
                paircnt[l][p] -= c
                if paircnt[l][p] <= 0: paircnt[l].pop(p, None)
                pairloc[l][p].discard(w)
            out, i, r = [], 0, 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b: out.append(ab); i += 2; r += 1
                else: out.append(sym[i]); i += 1
            reps[l][w] = out; tok_total[l] -= r * c
            for p in pairs(out): paircnt[l][p] += c; pairloc[l][p].add(w)
    while len(vocab) < V:
        cand = [l for l in langs if paircnt[l]]
        if not cand: break
        L = max(cand, key=fert)
        (a, b), _ = paircnt[L].most_common(1)[0]
        for l in langs:
            if (a, b) in pairloc[l]: apply_merge(l, a, b)
        vocab.add(a + b)
    return {l: fert(l) for l in langs}

if __name__ == "__main__":
    texts = load()
    print(f"loaded {len(texts)} languages\n")
    # whitespace vs scriptio-continua: show the \S+ word framework breaking for CJK
    print(f"{'lang':4}{'chars':>8}{'\\S+ words':>10}{'chars/word':>12}")
    for l in texts:
        w = re.findall(r"\S+", texts[l])
        print(f"{l:4}{len(texts[l]):8}{len(w):10}{len(texts[l])/max(1,len(w)):12.2f}"
              + ("   <- CJK: no spaces, metric breaks" if len(texts[l]) / max(1, len(w)) > 20 else ""))

    WS = [l for l in texts if len(texts[l]) / max(1, len(re.findall(r'\S+', texts[l]))) <= 20]  # whitespace langs
    def run(sel, V):
        f = parity_bpe({l: texts[l] for l in sel}, V)
        vals = list(f.values())
        print(f"  N={len(sel):2} V={V:6}: fertility {sum(vals)/len(vals):.3f}  "
              f"spread {max(vals)-min(vals):.4f}  [{min(vals):.3f}, {max(vals):.3f}]")
        return f

    print("\nPURE parity across growing, script-diverse WHITESPACE language sets:")
    run(["en", "hi", "te", "es"], 10000)                       # the original 4
    run(WS[:6], 10000)
    run(WS, 10000)                                             # all whitespace langs
    run(WS, 20000)                                             # more budget, same set
    print("\nper-language fertility, all whitespace langs @ V=10000 (is anyone starved?):")
    f = parity_bpe({l: texts[l] for l in WS}, 10000)
    for l in sorted(f, key=f.get): print(f"  {l}: {f[l]:.3f}")
