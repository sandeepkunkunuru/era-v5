#!/usr/bin/env python3
"""Build reversibility.ipynb — the notebook the assignment asks to submit.

It calls the same functions train.py does, so the notebook cannot disagree with run.log. The long
runs are replaced by short ones; the correctness checks and the reconstruction table run in full,
because those are the interesting part and they are cheap.

    python make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace reversibility.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-13-reversibility"
_ID = [0]


def _id():
    _ID[0] += 1
    return f"cell-{_ID[0]:02d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {},
            "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md("# ERA V5 · Session 13 — reversibility, measured",
       "",
       "A reversible stack throws its activations away in the forward pass and rebuilds them during",
       "the backward pass, so activation memory stops growing with depth. This notebook builds the",
       "three variants from the paper (arXiv:2512.02056) by hand, checks that their gradients match",
       "ordinary autograd, measures what the memory buys, and trains the three runs the assignment asks",
       "for at reduced length.",
       "",
       "Everything runs live. A GPU helps; the correctness checks run anywhere."),
    code("import os, sys, json",
         "if not os.path.exists('train.py'):              # on Colab: fetch the code",
         f"    !git clone -q {REPO} 2>/dev/null || true",
         f"    os.chdir('{DIR}')",
         "sys.path.insert(0, os.getcwd())",
         "import torch, train as T, model as M",
         "print('torch', torch.__version__, '| device', T.DEV)"),

    md("## The four stacks",
       "",
       "Only the rule carrying the hidden state between layers changes. `f` is the whole transformer",
       "block in every case (the paper's eq. 2.5).",
       "",
       "| stack | rule | reversible |",
       "|---|---|---|",
       "| baseline | `p[l+1] = p[l] + f(p[l])` | no — stores every layer's input |",
       "| midpoint | `p[l+1] = a·p[l-1] + 2h·f(p[l])` | yes |",
       "| leapfrog | `p[l+1] = 2p[l] - p[l-1] + h²·f(p[l])` | yes |",
       "| Hamiltonian | `q[l] = a·q[l-1] + Attn(LN(p[l-1]))`, `p[l] = a·p[l-1] + MLP(LN(q[l]))` | yes |",
       "",
       "Plain forward Euler is **not** reversible: undoing `p[l+1] = p[l] + h·f(p[l])` needs `f` at the",
       "state you are trying to recover."),

    md("## 1. Do the rebuilt gradients match ordinary autograd?",
       "",
       "The custom backward reconstructs each layer's input instead of remembering it. If that is right,",
       "the parameter gradients must equal what plain autograd produces on the same maths. In fp64:"),
    code("for r in T.grad_check():",
         "    print(f\"{r['stack']:12s} worst relative gradient error {r['worst_rel_grad_error']:.2e}\")"),

    md("## 2. Reconstruction, and what damping does to it",
       "",
       "V4's production integrator ran `a = 0.5`. The paper's stability analysis requires `|a| = 1` for a",
       "method to be stable forwards *and* backwards. The midpoint inverse divides by `a` at every layer,",
       "so a damped `a` multiplies any error on the way down."),
    code("for r in T.reconstruction_table():",
         "    print(f\"{r['stack']:12s} a={r['a']}  relative error {r['rel_error']:.3e}\")"),

    md("## 3. How much batch does the memory buy?",
       "",
       "The largest batch that survives two full steps, found by doubling then bisection."),
    code("caps = {s: T.max_batch(s) for s in ('baseline', 'midpoint')}",
         "print(caps, '→', f\"{caps['midpoint'] / caps['baseline']:.1f}x\")"),

    md("## 4. The three runs, shortened",
       "",
       "Baseline at a fixed batch, reversible at the same batch, reversible at its own maximum."),
    code("TOK = 400_000        # the full assignment uses 50M",
         "b = caps['baseline']",
         "rows = [T.run('baseline', b, TOK, '1-baseline'),",
         "        T.run('midpoint', b, TOK, '2-reversible-same-batch'),",
         "        T.run('midpoint', caps['midpoint'], TOK, '3-reversible-max-batch')]",
         "print()",
         "print(f\"{'run':26s} {'batch':>6s} {'tok/s':>10s} {'peak MiB':>10s} {'val loss':>9s}\")",
         "for r in rows:",
         "    print(f\"{r['tag']:26s} {r['batch']:6d} {r['tokens_per_s']:10,.0f} \"",
         "          f\"{r['peak_mem_mib']:10,.0f} {r['val_loss']:9.4f}\")"),

    md("## 5. Memory against depth",
       "",
       "The structural claim: the baseline carries a per-layer activation term and the reversible stack",
       "does not, so the gap widens as the stack gets deeper."),
    code("for r in T.memory_vs_depth(max(1, caps['baseline'] // 2), [4, 8, 12, 24]):",
         "    print(f\"{r['layers']:3d} layers  {r['stack']:9s} {r['peak_mem_mib']:8,.0f} MiB\")"),

    md("## What to take away",
       "",
       "- The reversible stacks reproduce ordinary autograd's gradients to floating-point noise, so the",
       "  memory saving costs nothing in correctness.",
       "- It is not free in time: rebuilding the activations re-runs every block, which the paper prices",
       "  at 30–50% and these runs confirm.",
       "- It only pays when the recovered memory is spent — on a bigger batch, a longer sequence, or a",
       "  deeper stack.",
       "- Dropout must be zero, or the rebuild draws a different mask and the gradients are wrong."),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "reversibility.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out.name} -- {len(CELLS)} cells")
