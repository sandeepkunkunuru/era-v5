#!/usr/bin/env python3
"""Build loss_harness.ipynb — the Colab-runnable version of the harness.

The notebook does not duplicate the code. It clones the repo and calls the same functions,
one requirement per cell, so a reader sees each answer on its own and the numbers cannot
drift between the notebook and `run.log`.

    python make_notebook.py
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-09-loss-harness"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md("# ERA V5 · Session 9 — the loss harness",
       "",
       "Everything between the model's output and the scalar the optimiser sees, made",
       "**observable**: shapes, the shift, padding, document boundaries, perplexity, weight",
       "tying, the memory cost of the loss itself — and then a second head predicting `t+2`.",
       "",
       "Every number below is computed live. Nothing is quoted from the lecture.",
       "",
       "Runs on a Colab T4 in a few minutes. A GPU is needed only for requirement 7 (peak",
       "memory) — everything else runs on CPU."),

    md("## Setup"),
    code("!pip -q install tiktoken",
        f"!git clone -q {REPO} 2>/dev/null || echo 'already cloned'",
        f"%cd {DIR}"),
    code("import math, torch, torch.nn.functional as F",
         "import loss_harness as H",
         "from model import GPT, Config",
         "",
         "torch.manual_seed(H.SEED)",
         "print('torch', torch.__version__, '| device', H.DEV)",
         "enc = H.get_tokenizer()",
         "V   = enc.n_vocab",
         "text = H.corpus()",
         "print(f'corpus {len(text):,} chars | GPT-2 BPE, V={V:,}')"),
    code("cfg = Config(vocab_size=V, block_size=256)",
         "torch.manual_seed(H.SEED)",
         "model = GPT(cfg).to(H.DEV).eval()",
         "ids = enc.encode(text[:20000])",
         "tokens = torch.tensor(ids[:4*cfg.block_size], device=H.DEV).view(4, cfg.block_size)",
         "print(f'{model.n_params():,} parameters')"),

    md("## Part 1 — the harness",
       "",
       "### 1. Every tensor shape, and what each dimension means"),
    code("H.req1(model, tokens, enc)"),

    md("### 2. Verify the shift by printing token **strings**",
       "",
       "Not ids. An off-by-one is invisible in a wall of integers and obvious in text."),
    code("H.req2(tokens, enc)"),

    md("### 3. Mask padding, and confirm the contributing-token count changes",
       "",
       "Also covers the denominator bug, and why `<eos>` must *not* be masked the way `<pad>` is."),
    code("H.req3(model, enc, V)"),

    md("### 4. Pack two documents into one sequence, and mask the boundary",
       "",
       "Shakespeare next to Python source, so the join is as unrelated as real packing makes it."),
    code("H.req4(model, enc, V, text)"),

    md("### 5. Perplexity, and where an untrained model must start"),
    code("H.req5(model, tokens, V)"),

    md("### 6. Tied against untied output head"),
    code("H.req6(cfg)"),

    md("### 7. Peak memory: ordinary cross-entropy against a chunked one",
       "",
       "**Needs a GPU.** Same objective, two implementations — the loss must not move."),
    code("H.req7(model, V)   # note: frees `model`"),

    md("## Part 2 — a second head at `t+2`",
       "",
       "Trains a small model with heads at `t+1` and `t+2`, whose losses add. Reports both",
       "separately and their sum, and then chases the one thing in the curve that should not",
       "be there — with two controls, both of which refute the explanation they were testing."),
    code("H.part2(text, enc, V, steps=1500, log_every=150)"),

    md("## Results",
       "",
       "`results.json` holds every number above in machine-readable form; `run.log` is a full",
       "transcript of one run. The write-up is in `README.md`."),
    code("import json; print(json.dumps(H.R, indent=2)[:2000], '...')"),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).parent / "loss_harness.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} — {len(CELLS)} cells")
