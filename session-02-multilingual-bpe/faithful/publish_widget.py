#!/usr/bin/env python3
"""Publish the faithful tokenizer + data into the live Session-2 widget (../../session-2/).

Copies tokenizer.json / metrics.json, writes tokens.txt (vocab by id) and a page-shaped results.json,
and refreshes data/{en,hi,te,mai}.txt (the faithful corpus the widget re-tokenizes live). The page
markup (index.html, css, js) is hand-authored and versioned separately; this only refreshes the data.

    python publish_widget.py
"""
import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DST = (HERE / "../../session-2").resolve()
LANGS = ["en", "hi", "te", "mai"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"}
SCRIPTS = {"en": "Latin", "hi": "Devanagari", "te": "Telugu", "mai": "Devanagari"}
TITLES = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "mai": "भारत"}

metrics = json.loads((HERE / "metrics.json").read_text())
(DST / "data").mkdir(parents=True, exist_ok=True)

# artifacts
shutil.copyfile(HERE / "tokenizer.json", DST / "tokenizer.json")
shutil.copyfile(HERE / "metrics.json", DST / "metrics.json")

# tokens.txt — vocab ordered by id
tok = Tokenizer.from_file(str(HERE / "tokenizer.json"))
toks = [t for t, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1])]
(DST / "tokens.txt").write_text("\n".join(toks), encoding="utf-8")

# corpus the widget re-tokenizes live
for l in LANGS:
    shutil.copyfile(HERE / f"corpus/{l}.faithful.txt", DST / f"data/{l}.txt")
# drop the stale Spanish corpus from the old build
old_es = DST / "data/es.txt"
if old_es.exists():
    old_es.unlink()

# page-shaped results.json
ratios = metrics["ratios"]
xs = sorted(ratios.values())
results = {
    "method": "faithful BPE — HF BPE, vocab 10k, NFKC, Metaspace pre-tok+decoder, weights en3/hi4/te4/mai2",
    "langs": LANGS,
    "names": NAMES,
    "scripts": SCRIPTS,
    "titles": TITLES,
    "vocab": metrics["vocab_size"],
    "faithful_units": metrics["faithful_units"],
    "token_counts": metrics["token_counts"],
    "fertility": ratios,
    "X1": xs[0],
    "X4": xs[-1],
    "spread": metrics["spread"],
    "score": metrics["score"],
    "cap": 1.2,
    "all_below_cap": all(v <= 1.2 for v in ratios.values()),
}
(DST / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"published to {DST}")
print(f"  score {results['score']:.0f}  spread {results['spread']:.4f}  vocab {results['vocab']}")
print(f"  fertility: " + "  ".join(f"{l} {ratios[l]:.4f}" for l in LANGS))
print(f"  data: {', '.join(l+'.txt' for l in LANGS)} (es.txt removed)")
