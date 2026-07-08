"""H4 — H3 script-disjoint union + whole-word budget reallocation (Picky-BPE / BPE-knockout idea).

Diagnosis behind H4: the H3 tokenizer (score 2,336) leaves ~17% of its 10,000 slots on intermediate
BPE merges that never surface as final tokens, while whole-word coverage stays low (Hindi 21%,
Spanish 17%). Published work — Picky BPE (Chizhov et al., EMNLP 2024) and BPE-knockout (Bauwens &
Delobelle, NAACL 2024) — shows those wasted slots can be reclaimed. H4 does exactly that, tuned for
this fertility-parity objective:

  1. Train the script-disjoint union (joint Latin + disjoint Devanagari + Telugu) at a REDUCED
     budget, so fewer slots go to intermediates.
  2. Spend the freed slots on explicit whole-word merges: repeatedly take the currently
     highest-fertility language and collapse its most frequent multi-token word to a single token
     (append the merge of its first adjacent pair; multi-token words collapse fully over rounds).

Still ONE valid BPE tokenizer (vocab + ordered merges) with a WhitespaceSplit pre-tokenizer, so the
grader can re-run it. English stays at 1.180 — a safe margin under the 1.2 cap.  Result: gap 0.41,
score ~2,430 (vs H3's 0.428 / 2,336).
"""
import os, re, json
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/data"
LANGS = ["en", "hi", "te", "es"]; VOCAB = 10000; EN_CAP = 1.2
# reduced base budget (leaves ~1,600 slots to reallocate); attack these languages with whole words
LAT_B, W_EN, HI_B, TE_B = 6300, 4, 1100, 2200
TARGETS = ["es", "hi", "te"]
PAGE_TITLES = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "es": "India"}

RAW = {l: open(f"{DATA}/{l}.txt", encoding="utf-8").read() for l in LANGS}
WORDS = {l: len(re.findall(r"\S+", RAW[l])) for l in LANGS}
FREQ = {l: Counter(re.findall(r"\S+", RAW[l])) for l in LANGS}
LINES = {l: [ln for ln in RAW[l].split("\n") if ln.strip()] for l in LANGS}

def _train(iter_lines, budget):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=budget, special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(iter_lines, tr)
    d = json.loads(tok.to_str())["model"]
    return d["vocab"], [tuple(m) for m in d["merges"]]

def latin_group(budget, w_en):
    def corpus():
        for _ in range(w_en):
            for ln in LINES["en"]: yield ln
        for ln in LINES["es"]: yield ln
    return _train(corpus(), budget)

def make(uv, mseq):
    tok = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in mseq], unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok

def evaluate(tok):
    fert = {l: len(tok.encode(RAW[l]).tokens) / WORDS[l] for l in LANGS}
    xs = sorted(fert.values())
    return fert, xs[0], xs[-1]

# --- 1. base script-disjoint union at reduced budget -----------------------------------------
uv, seen, mseq = {}, set(), []
def add(t):
    if t not in uv: uv[t] = len(uv)
add("[UNK]")
for vocab, merges in [latin_group(LAT_B, W_EN), _train(LINES["hi"], HI_B), _train(LINES["te"], TE_B)]:
    for t in sorted(vocab, key=lambda x: vocab[x]): add(t)
    for ab in merges:
        if ab not in seen: seen.add(ab); mseq.append(ab)
base_size = len(uv)
tok = make(uv, mseq)
print(f"base union: {base_size} tokens  ({VOCAB - base_size} slots to reallocate)")
bf, *_ = evaluate(tok)
print(f"  before reallocation: en {bf['en']:.3f} hi {bf['hi']:.3f} te {bf['te']:.3f} es {bf['es']:.3f}")

# --- 2. reallocate freed slots to whole words for the worst languages ------------------------
order = {l: [w for w, _ in FREQ[l].most_common()] for l in TARGETS}
targets = list(TARGETS); rounds = 0
while len(uv) < VOCAB and rounds < 80:
    rounds += 1
    fert, *_ = evaluate(tok)
    lang = max(targets, key=lambda l: fert[l])           # attack the current worst language
    added, limit = 0, min(120, VOCAB - len(uv))
    for w in order[lang]:
        if added >= limit or len(uv) >= VOCAB: break
        enc = tok.encode(w).tokens
        if len(enc) < 2: continue
        a, b = enc[0], enc[1]; prod = a + b               # collapse the first adjacent pair
        if prod in uv: continue
        uv[prod] = len(uv); mseq.append((a, b)); added += 1
    if added == 0:
        targets = [l for l in targets if l != lang]
        if not targets: break
        continue
    tok = make(uv, mseq)

# --- 3. finalize + verify --------------------------------------------------------------------
fert, x1, x4 = evaluate(tok); gap = x4 - x1; sc = 1000 / gap
assert fert["en"] <= EN_CAP, f"English fertility {fert['en']:.4f} exceeds cap {EN_CAP}"
assert len(uv) <= VOCAB, f"vocab {len(uv)} exceeds {VOCAB}"
print(f"\nfinal ({len(uv)} tokens): en {fert['en']:.3f} hi {fert['hi']:.3f} te {fert['te']:.3f} "
      f"es {fert['es']:.3f}  gap {gap:.3f}  SCORE {sc:.1f}")

tok.save(f"{HERE}/tokenizer.json")
vocab = tok.get_vocab()
toks = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
open(f"{HERE}/tokens.txt", "w", encoding="utf-8").write("\n".join(toks))
json.dump({"method": "H4-reallocation (H3 union + whole-word budget reclamation)", "langs": LANGS,
           "vocab": len(toks), "en_cap": EN_CAP,
           "alloc": {"latin_budget": LAT_B, "en_weight": W_EN, "hi_budget": HI_B, "te_budget": TE_B,
                     "reallocated_slots": len(uv) - base_size, "targets": TARGETS},
           "words": WORDS, "fertility": fert, "X1": x1, "X4": x4, "gap": gap, "score": sc,
           "page_titles": PAGE_TITLES},
          open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=2)
print(f"saved tokenizer.json, tokens.txt ({len(toks)} tokens), results.json")
