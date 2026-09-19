#!/usr/bin/env python3
"""Build optimizers.ipynb -- the notebook version of the harness.

It calls the same functions as optimizers.py, so the numbers cannot drift from run.log. It runs with fewer
steps and grid points than the full run so it finishes in a few minutes on a GPU; the full numbers are in
README.md. Figures go to figures/notebook/ so the full run's figures are never overwritten.

    python make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace optimizers.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-11-optimizers"
_ID = [0]


def _id():
    _ID[0] += 1
    return f"cell-{_ID[0]:02d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {}, "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md("# ERA V5 · Session 11 — optimizers and learning-rate schedules, measured",
       "",
       "Every number below is computed live. The full run (300 steps, 8-point sweeps) is in `run.log` and `README.md`;",
       "this notebook uses shorter runs so it finishes in a few minutes."),
    code("import os, sys",
         "if not os.path.exists('optimizers.py'):              # on Colab: fetch the code",
         "    !pip -q install tiktoken",
         f"    !git clone -q {REPO} 2>/dev/null || true",
         f"    os.chdir('{DIR}')",
         "sys.path.insert(0, os.getcwd())",
         "import torch, optimizers as H",
         "from IPython.display import Image, display",
         "import matplotlib; matplotlib.use('Agg')",
         "_save = H._save",
         "def _save_nb(fig, name):                              # keep the full run's figures untouched",
         "    return _save(fig, 'notebook_' + name)",
         "H._save = _save_nb",
         "print('torch', torch.__version__, '| device', H.DEV)",
         "train_ids, val = H.token_splits()"),
    md("## 1. Adam by hand — one weight, five gradients, checked against PyTorch"),
    code("r1 = H.req1()"),
    md("## 2. Bias correction switched off"),
    code("r2 = H.req2(train_ids, val, steps=120)", "display(Image(filename=r2['figure']))"),
    md("## 3. The update-to-weight ratio, every layer"),
    code("r3 = H.req3(train_ids, val, steps=160, warm=60)", "display(Image(filename=r3['figure']))"),
    md("## 4. Cosine against WSD, both tuned, stopped at step 200"),
    code("r4 = H.req4(train_ids, val, quick=True)", "display(Image(filename=r4['figure']))"),
    md("## 5. Learning-rate sweep at widths 256, 512 and 1,024",
       "",
       "`quick=True` uses 3 learning rates and 120 steps; the full 8-point, 300-step sweep is in the README."),
    code("r5 = H.req5(train_ids, val, quick=True)", "display(Image(filename=r5['figure']))"),
]
nb = {"cells": CELLS,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "optimizers.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out.name} -- {len(CELLS)} cells")
