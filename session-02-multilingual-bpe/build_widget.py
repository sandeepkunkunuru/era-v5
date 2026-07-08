"""Publish the trained artifacts into the Session 2 page (../session-2/).

The page itself (index.html, css/app.css, js/app.js) is hand-authored and versioned in
../session-2/. This script only refreshes the data it serves: after re-running train_h3.py,
run this to copy the tokenizer, token list, results, and the four page texts into session-2/
so the live number-line, downloads, and in-browser verification pick up the new numbers.

Usage:  ~/projects/venv/bin/python build_widget.py
"""
import os, shutil, json

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.abspath(f"{HERE}/../session-2")
LANGS = ["en", "hi", "te", "es"]

os.makedirs(f"{DST}/data", exist_ok=True)
for name in ["tokenizer.json", "tokens.txt", "results.json"]:
    shutil.copyfile(f"{HERE}/{name}", f"{DST}/{name}")
for l in LANGS:
    shutil.copyfile(f"{HERE}/data/{l}.txt", f"{DST}/data/{l}.txt")

r = json.load(open(f"{DST}/results.json", encoding="utf-8"))
print(f"published to {DST}")
print(f"  score {r['score']:.0f}  gap {r['gap']:.3f}  en {r['fertility']['en']:.3f}  "
      f"vocab {r['vocab']}  ({', '.join(name for name in os.listdir(DST) if not name.startswith('.'))})")
