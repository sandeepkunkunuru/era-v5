#!/usr/bin/env python3
"""Build zero_sim.ipynb — the notebook version of the simulation.

It calls the same functions as zero_sim.py, so the numbers cannot drift from run.log. The notebook runs
all four stages at 32 virtual GPUs with fewer steps than the full run; the 8- and 16-GPU runs are in the
README. Figures go to figures/notebook/ so the full run's figures are never overwritten.

    python make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace zero_sim.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-12-zero"


_ID = [0]


def _id():
    _ID[0] += 1
    return f"cell-{_ID[0]:02d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {}, "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md("# ERA V5 · Session 12 — ZeRO on 32 virtual GPUs, by hand",
       "",
       "32 CPU processes stand in for 32 GPUs and talk through `torch.distributed` (gloo) with the same three",
       "collectives a real run uses: all-reduce, reduce-scatter, all-gather. A tiny GPT trains on top of them",
       "under data parallelism (stage 0) and ZeRO stages 1, 2 and 3. No DeepSpeed, no FSDP: each stage is the",
       "collectives called in the right order, in `zero_sim.py`.",
       "",
       "Everything below runs live. Runs on CPU; no GPU needed."),
    code("import os, sys, json",
         "if not os.path.exists('zero_sim.py'):             # on Colab: fetch the code",
         f"    !git clone -q {REPO} 2>/dev/null || true",
         f"    os.chdir('{DIR}')",
         "sys.path.insert(0, os.getcwd())",
         "import torch, zero_sim as Z",
         "Z.FIGDIR = Z.HERE / 'figures' / 'notebook'       # keep the full run's figures untouched",
         "Z.load_text()",
         "print('torch', torch.__version__, '| CPU threads', os.cpu_count())"),

    md("## What each stage keeps on every GPU",
       "",
       "| stage | weights (bf16, 2 B) | gradients (bf16, 2 B) | fp32 copy + Adam m, v (12 B) | bytes / parameter at W GPUs |",
       "|---|---|---|---|---|",
       "| 0 (data parallel) | all | all | all | 16 |",
       "| 1 | all | all | 1/W | 4 + 12/W |",
       "| 2 | all | 1/W | 1/W | 2 + 14/W |",
       "| 3 | 1/W | 1/W | 1/W | 16/W |",
       "",
       "The run below measures these by walking the tensors each process actually holds."),

    md("## Run all four stages on 32 virtual GPUs"),
    code("W, STEPS = 32, 6",
         "runs = []",
         "for stage in range(4):",
         "    s = Z.summarise(Z.run(W, stage, STEPS, 29800 + stage))",
         "    runs.append(s)",
         "    print(f\"stage {stage}: {s['bytes_per_param']:5.2f} B/param per GPU (formula {Z.theory(stage, W):5.2f}) | \"",
         "          f\"traffic {s['traffic_total_P']:.3f} P/step | optimizer {1000*s['time']['optimizer']:.2f} ms | \"",
         "          f\"loss after {STEPS} steps {s['losses'][-1]:.4f}\")"),

    md("## Did the sharding change what the model learned?",
       "",
       "One process, trained on the whole global batch of 32 x 4 sequences. All five loss curves must agree."),
    code("ref = Z.reference(STEPS, W)",
         "print('step  one-process  ' + '  '.join(f'stage {s[\"stage\"]}' for s in runs))",
         "for i in range(STEPS):",
         "    print(f'{i+1:>4}  {ref[i]:11.4f}  ' + '  '.join(f'{s[\"losses\"][i]:7.4f}' for s in runs))",
         "print('largest difference:', {s['stage']: f\"{max(abs(a - b) for a, b in zip(s['losses'], ref)):.1e}\" for s in runs})"),

    md("## Memory and time per GPU"),
    code("from IPython.display import Image, display",
         "R = dict(worlds=[W], runs=runs, N=runs[0]['N'], cpu_threads=os.cpu_count())",
         "Z.plot_all(R)",
         "display(Image(filename=str(Z.FIGDIR / 'memory_by_stage.png')))",
         "display(Image(filename=str(Z.FIGDIR / 'time_by_stage.png')))"),

    md("## Communication, collective by collective",
       "",
       "In units of P, the whole model in bf16. The ring volumes are 2(W-1)/W for all-reduce and (W-1)/W for",
       "reduce-scatter and all-gather, so stages 0-2 come to 2(W-1)/W and stage 3 to just under 3(W-1)/W",
       "(the embeddings need no weights in backward, so they skip one gather)."),
    code("for s in runs:",
         "    print(f\"stage {s['stage']}: \", {k: round(v, 3) for k, v in s['traffic_P'].items()}, '| calls', s['calls'])"),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "zero_sim.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out.name} -- {len(CELLS)} cells")
