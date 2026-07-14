#!/usr/bin/env python3
"""Build the corpus for the vocab-size fertility sweep (Session 3, India-first 40B).

TRAIN corpus (to train candidate tokenizers): Indic-heavy, mirrors the intended India-first mix —
Wikipedia text for en + 12 Indic langs, plus code (GitHub raw) and math (open-web-math).
EVAL corpus (to measure fertility fairly): FLORES-200 devtest (parallel: identical semantic content
across all 13 languages, so fertility differences are tokenization/script, not content) + held-out
code + math.

    python fetch_corpus.py
"""
import sys, time, pathlib, urllib.request
from datasets import load_dataset

ROOT = pathlib.Path(__file__).resolve().parent
TRAIN = ROOT / "data/train"; EVAL = ROOT / "data/eval"
TRAIN.mkdir(parents=True, exist_ok=True); EVAL.mkdir(parents=True, exist_ok=True)

# 13 target languages: wiki config code -> FLORES devtest code
LANGS = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "mr": "mar_Deva",
    "te": "tel_Telu", "ta": "tam_Taml", "gu": "guj_Gujr", "kn": "kan_Knda",
    "ml": "mal_Mlym", "pa": "pan_Guru", "or": "ory_Orya", "ur": "urd_Arab", "as": "asm_Beng",
}
WIKI_MB = 3.0   # per language, training text

def fetch_wiki(lang, cap_bytes):
    out = TRAIN / f"wiki_{lang}.txt"
    if out.exists() and out.stat().st_size >= cap_bytes * 0.8:
        print(f"  wiki {lang}: cached {out.stat().st_size/1e6:.1f}MB"); return
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    got = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            t = row["text"].strip()
            if len(t) < 200: continue
            f.write(t + "\n\n"); got += len(t.encode("utf-8"))
            if got >= cap_bytes: break
    print(f"  wiki {lang}: {got/1e6:.1f}MB")

# code: ungated GitHub raw files across languages (the-stack-smol is gated)
CODE_URLS = [
    "https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/functools.py",
    "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py",
    "https://raw.githubusercontent.com/psf/requests/main/src/requests/models.py",
    "https://raw.githubusercontent.com/lodash/lodash/main/lodash.js",
    "https://raw.githubusercontent.com/facebook/react/main/packages/react/src/ReactHooks.js",
    "https://raw.githubusercontent.com/golang/go/master/src/net/http/server.go",
    "https://raw.githubusercontent.com/rust-lang/rust/master/library/alloc/src/vec/mod.rs",
    "https://raw.githubusercontent.com/torvalds/linux/master/kernel/sched/core.c",
    "https://raw.githubusercontent.com/microsoft/TypeScript/main/src/compiler/checker.ts",
    "https://raw.githubusercontent.com/apache/spark/master/sql/core/src/main/scala/org/apache/spark/sql/Dataset.scala",
    "https://raw.githubusercontent.com/postgres/postgres/master/src/backend/parser/gram.y",
]
def fetch_code():
    train_f = TRAIN / "code.txt"; eval_f = EVAL / "code.txt"
    if train_f.exists() and eval_f.exists():
        print(f"  code: cached"); return
    texts = []
    for u in CODE_URLS:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "era-v5-s3/1.0"})
            texts.append(urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace"))
        except Exception as e:
            print(f"    code skip {u.split('/')[-1]}: {repr(e)[:60]}")
    split = max(1, int(len(texts) * 0.8))
    train_f.write_text("\n\n".join(texts[:split]), encoding="utf-8")
    eval_f.write_text("\n\n".join(texts[split:]), encoding="utf-8")
    print(f"  code: train {train_f.stat().st_size/1e6:.1f}MB / eval {eval_f.stat().st_size/1e6:.1f}MB")

def fetch_math(train_mb=4.0, eval_mb=1.0):
    train_f = TRAIN / "math.txt"; eval_f = EVAL / "math.txt"
    if train_f.exists() and eval_f.exists():
        print(f"  math: cached"); return
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    tr, ev, tb, eb = [], [], 0, 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200: continue
        if tb < train_mb * 1e6: tr.append(t); tb += len(t.encode("utf-8"))
        elif eb < eval_mb * 1e6: ev.append(t); eb += len(t.encode("utf-8"))
        else: break
    train_f.write_text("\n\n".join(tr), encoding="utf-8")
    eval_f.write_text("\n\n".join(ev), encoding="utf-8")
    print(f"  math: train {tb/1e6:.1f}MB / eval {eb/1e6:.1f}MB")

def link_flores():
    src = ROOT / "data/flores200_dataset/devtest"
    for lang, fc in LANGS.items():
        f = src / f"{fc}.devtest"
        if f.exists():
            (EVAL / f"flores_{lang}.txt").write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            print(f"    FLORES MISSING {fc}")
    print(f"  flores eval: {len(list(EVAL.glob('flores_*.txt')))} langs")

if __name__ == "__main__":
    print("wikipedia (train):")
    for lang in LANGS:
        try: fetch_wiki(lang, WIKI_MB * 1e6)
        except Exception as e: print(f"  wiki {lang} FAIL: {repr(e)[:80]}")
    print("code (train+eval):"); fetch_code()
    print("math (train+eval):"); fetch_math()
    print("flores (eval):"); link_flores()
    print("done.")
