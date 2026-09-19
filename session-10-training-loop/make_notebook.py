#!/usr/bin/env python3
"""Build training_loop.ipynb — the notebook version of the harness.

The notebook does not duplicate the code. It calls the same functions as training_loop.py,
one requirement per cell, so the numbers cannot drift between the notebook and run.log.
It runs locally or on Colab (the first cell clones the repo only if the code is not there),
with fewer training steps than the full run so it finishes in a few minutes.

    python make_notebook.py                 # write training_loop.ipynb (no outputs)
    jupyter nbconvert --to notebook --execute --inplace training_loop.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-10-training-loop"


_ID = [0]


def _id():
    _ID[0] += 1
    return f"cell-{_ID[0]:02d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {}, "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md("# ERA V5 · Session 10 — making a training loop tell the truth",
       "",
       "A small GPT and a real loop, instrumented until every claim can be checked rather than believed.",
       "Every number below is computed live in this notebook. The full-length run (400 steps, plus the",
       "`torch.compile` MFU variants) is in `run.log` and `README.md`; this notebook uses fewer steps."),
    md("## Setup"),
    code("import os, sys",
         "if not os.path.exists('training_loop.py'):          # on Colab: fetch the code",
         "    !pip -q install tiktoken",
         f"    !git clone -q {REPO} 2>/dev/null || true",
         f"    os.chdir('{DIR}')",
         "sys.path.insert(0, os.getcwd())",
         "import torch",
         "import training_loop as H",
         "H.FIGDIR = H.HERE / 'figures' / 'notebook'   # keep the full run's figures untouched",
         "print('torch', torch.__version__, '| device', H.DEV,",
         "      '|', torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')",
         "train_ids, val = H.token_splits()",
         "print(f'{len(train_ids):,} train tokens, {len(val):,} validation tokens')"),

    md("## 1. Every tensor shape in one step, and what each dimension means",
       "",
       "Block 0 is unrolled by hand, attention included, and asserted equal to the module's output."),
    code("r1 = H.req1(train_ids)"),

    md("## 2. One gradient, verified by hand",
       "",
       "Nudge one weight, measure the loss, compare the slope with `backward()`. The first attempt uses the",
       "model as Session 9 left it and does **not** agree — the cell explains why, then checks again."),
    code("r2 = H.req2(train_ids)"),

    md("## 3. Gradient accumulation, broken on purpose",
       "",
       "Micro-batches of unequal length. Correct: sum of token losses over all tokens. Buggy: average of the",
       "micro-batch averages. Same initialisation, same data."),
    code("r3 = H.req3(train_ids, val, steps=150)",
         "from IPython.display import Image, display",
         "display(Image(filename=r3['figure']))"),

    md("## 4. The grad norm every step — and a step where it moved before the loss"),
    code("r4 = H.req4(train_ids, steps=250)",
         "if r4['found']: display(Image(filename=r4['figure']))"),

    md("## 5. MFU",
       "",
       "Measured against this GPU's own bf16 matmul peak. `quick=True` skips the `torch.compile` variants;",
       "the full table, including the 4x wider model, is in `README.md`."),
    code("r5 = H.req5(train_ids, quick=True)"),

    md("## 6. 0.1 in fp32, bf16 and fp8 E4M3, bit by bit",
       "",
       "Exact arithmetic, round to nearest even, then checked against the bits the real formats produce."),
    code("r6 = H.req6()"),
]

nb = {"cells": CELLS,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "training_loop.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out.name} -- {len(CELLS)} cells")
