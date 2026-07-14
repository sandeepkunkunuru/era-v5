#!/usr/bin/env python3
"""Vocab-size fertility sweep for the India-first 40B tokenizer decision.

Trains an Indic-aware char-level BPE (NFKC + Metaspace, like Sarvam/Gemma's SentencePiece) at several
vocab sizes on the Indic-heavy mixed TRAIN corpus, then measures fertility on held-out EVAL sets:
  - languages : tokens / word on FLORES-200 devtest (parallel -> fair cross-lingual comparison),
                plus the Petrov-style "premium vs English".
  - code/math : tokens per 1000 chars (script-agnostic).
The point: find where per-language fertility stops improving (the knee) and where Indic reaches
acceptable parity with English -> the defensible vocab number for a 40B India-first model.

    python sweep_vocab.py
Writes results.json (per vocab size) incrementally.
"""
import json, glob, os, pathlib, tempfile
import regex
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

ROOT = pathlib.Path(__file__).resolve().parent
TRAIN = ROOT / "data/train"; EVAL = ROOT / "data/eval"
OUT = ROOT / "results.json"
VOCAB_SIZES = [32000, 64000, 128000, 200000, 256000]
LANGS = ["en","hi","bn","mr","te","ta","gu","kn","ml","pa","or","ur","as"]
WORD = regex.compile(r"\S+")

def train_files():
    return sorted(glob.glob(str(TRAIN / "*.txt")))

def make_tok():
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.normalizer = normalizers.NFKC()
    t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
    t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
    return t

def fertility_words(tok, text):
    words = WORD.findall(text)
    if not words: return None
    return len(tok.encode(text).ids) / len(words)

def toks_per_kchar(tok, text):
    return 1000.0 * len(tok.encode(text).ids) / max(1, len(text))

def main():
    files = train_files()
    assert files, "no training files — run fetch_corpus.py first"
    print(f"training on {len(files)} files, {sum(os.path.getsize(f) for f in files)/1e6:.0f}MB")
    evals = {l: (EVAL / f"flores_{l}.txt").read_text(encoding="utf-8") for l in LANGS}
    code = (EVAL / "code.txt").read_text(encoding="utf-8")
    math = (EVAL / "math.txt").read_text(encoding="utf-8")

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for V in VOCAB_SIZES:
        key = str(V)
        if key in results:
            print(f"vocab {V}: cached"); continue
        tok = make_tok()
        tok.train(files, BpeTrainer(vocab_size=V, min_frequency=2, special_tokens=["[UNK]"]))
        fert = {l: fertility_words(tok, evals[l]) for l in LANGS}
        en = fert["en"]
        row = {
            "vocab": V,
            "fertility": fert,
            "premium_vs_en": {l: fert[l] / en for l in LANGS},
            "code_tok_per_kchar": toks_per_kchar(tok, code),
            "math_tok_per_kchar": toks_per_kchar(tok, math),
            "indic_mean": sum(fert[l] for l in LANGS if l != "en") / (len(LANGS) - 1),
            "indic_spread": max(fert[l] for l in LANGS if l != "en") - min(fert[l] for l in LANGS if l != "en"),
            "actual_vocab": tok.get_vocab_size(),
        }
        results[key] = row
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"vocab {V:>7}: en {en:.2f} | indic mean {row['indic_mean']:.2f} "
              f"(max premium {max(row['premium_vs_en'].values()):.2f}x) | "
              f"code {row['code_tok_per_kchar']:.0f}/kc math {row['math_tok_per_kchar']:.0f}/kc")
    print("done ->", OUT)

if __name__ == "__main__":
    main()
