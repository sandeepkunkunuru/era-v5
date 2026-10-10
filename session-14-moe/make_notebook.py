#!/usr/bin/env python3
"""Build moe.ipynb — a live, short version of the assignment.

It imports the same `model.py` and `train.py` the real run used, so the notebook cannot disagree
with run.log. Token budgets are cut to minutes; the conversion checks run in full, because they are
exact and cheap.

    python make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace moe.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/sandeepkunkunuru/era-v5.git"
DIR = "era-v5/session-14-moe"
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
    md("# ERA V5 · Session 14 — a dense model grown into a mixture of experts",
       "",
       "> Train a Linear model and convert that into an MoE! Your call on model size and data trained",
       "> on, but must show they continue to train and reduce loss!",
       "",
       "This notebook runs a short version of the real run live: train a dense byte-level transformer,",
       "convert it into a mixture of experts four different ways, show the conversion can be **exact**,",
       "and then keep training. The full-length numbers are in `README.md` and `results.json`; the",
       "training log the assignment asks for is `run.log`.",
       "",
       "A GPU helps. The conversion checks run anywhere."),

    code("import os, sys, json",
         "if not os.path.exists('train.py'):              # on Colab: fetch the code",
         f"    !git clone -q {REPO} 2>/dev/null || true",
         f"    os.chdir('{DIR}')",
         "sys.path.insert(0, os.getcwd())",
         "import torch, train as T, model as M",
         "print('torch', torch.__version__, '| device', T.DEV)"),

    md("## The two shapes",
       "",
       "The mixture-of-experts model is built so that each token uses **exactly the dense model's",
       "feed-forward width**. The dense inner width is 1,024; the experts are 128 wide and 8 are chosen,",
       "and 8 x 128 = 1,024. So the only thing that changes is how many feed-forward parameters the layer",
       "*stores*."),
    code("dense = M.GPT(M.Config())",
         "moe = M.GPT(M.Config(moe=True))",
         "d, m = dense.param_counts(), moe.param_counts()",
         "print(f\"dense  {d['total']:>12,} total  {d['active']:>12,} active\")",
         "print(f\"moe    {m['total']:>12,} total  {m['active']:>12,} active\")",
         "print(f\"ratio  {m['total']/d['total']:>12.2f}x {m['active']/d['active']:>12.2f}x\")"),

    md("## 1. Train the dense model",
       "",
       "Short here — 1M tokens of Session 4's cleaned corpus, as bytes. The real run used 12M."),
    code("row = T.run(dense, 'nb-dense', 1_000_000, batch=8, accum=1)",
         "dense_val = row['val_loss']",
         "dense = dense.cpu()",
         "print('dense validation loss', dense_val)"),

    md("## 2. The conversion can be exact",
       "",
       "Force the router onto the inherited experts and scale the down-projection by top-k. Then the",
       "mixture-of-experts layer *is* the dense layer: `partition` and `clone` hold exactly the dense",
       "network's neurons between them, and a weighted sum whose weights add to one, times k, puts it",
       "back together. `drop` cannot, because half of each expert is new."),
    code("va = T.Batches('val', 8)",
         "for meth in ('clone', 'partition', 'drop'):",
         "    net = M.upcycle(dense, meth, M.Config(moe=True, balance='bias'), scale_down=True).to(T.DEV)",
         "    for lay in net.moe_layers():",
         "        torch.nn.init.zeros_(lay.router.weight); lay.bias.zero_(); lay.bias[:8] = 1.0",
         "    v = T.evaluate(net, va)",
         "    print(f'{meth:10s} val {v:.6f}   delta {v - dense_val:+.6f}')",
         "    del net; torch.cuda.empty_cache() if T.DEV == 'cuda' else None"),

    md("## 3. The scaling factor is not a detail",
       "",
       "With a real router the k kept weights are each about 1/k, so the layer's output is about 1/k of",
       "the dense layer's. The brief calls the correction a *routed scaling factor* (2.5 in DeepSeek-V3,",
       "1.5 in V4.1-Flash) and notes the papers give the value without explaining it. Here the right",
       "value is known exactly — it is top-k."),
    code("for meth in ('random', 'clone', 'partition', 'drop'):",
         "    for sc in (False, True):",
         "        net = M.upcycle(dense, meth, M.Config(moe=True), scale_down=sc).to(T.DEV)",
         "        v = T.evaluate(net, va)",
         "        print(f'{meth:10s} scale_down={str(sc):5s} val {v:7.4f}  delta {v - dense_val:+.4f}')",
         "        del net; torch.cuda.empty_cache() if T.DEV == 'cuda' else None"),

    md("## 4. Does it keep training, and does the loss drop?",
       "",
       "The assignment's actual question. Each grown model gets 600k further tokens, against a dense",
       "control given the same."),
    code("ctrl = M.GPT(M.Config()); ctrl.load_state_dict(dense.state_dict())",
         "rows = [T.run(ctrl, 'nb-dense-control', 600_000, batch=8, accum=1)]",
         "del ctrl; torch.cuda.empty_cache() if T.DEV == 'cuda' else None",
         "for meth in ('random', 'partition'):",
         "    net = M.upcycle(dense, meth, M.Config(moe=True, balance='bias'),",
         "                    scale_down=(meth != 'random'))",
         "    rows.append(T.run(net, f'nb-{meth}', 600_000, batch=8, accum=1))",
         "    del net; torch.cuda.empty_cache() if T.DEV == 'cuda' else None",
         "for r in rows:",
         "    print(f\"{r['tag']:18s} val {r['val_loss']:7.4f}  {r['tokens_per_s']:>7,} tok/s\"",
         "          f\"  MaxVio {r.get('maxvio','--')}  dead {r.get('dead','--')}\")"),

    md("## 5. Balancing is what keeps experts alive",
       "",
       "Same init, same tokens, balancing off against the auxiliary-loss-free bias. Watch the",
       "dead-expert count, not the loss: an unused expert costs nothing today and costs its whole",
       "capacity for the rest of training."),
    code("for bal in ('none', 'bias'):",
         "    net = M.upcycle(dense, 'partition', M.Config(moe=True, balance=bal), scale_down=True)",
         "    r = T.run(net, f'nb-balance-{bal}', 600_000, batch=8, accum=1)",
         "    print(f\"{bal:6s} val {r['val_loss']:.4f}  MaxVio {r['maxvio']}\"",
         "          f\"  worst layer {r['maxvio_worst']}  dead {r['dead']}/{r['dead_of']}\")",
         "    del net; torch.cuda.empty_cache() if T.DEV == 'cuda' else None"),

    md("## 6. The full-length numbers",
       "",
       "Read from `results.json`, which came from `python train.py`."),
    code("R = json.loads(open('results.json').read())",
         "print('dense      ', {k: R['dense'][k] for k in ('tokens', 'val_loss', 'tokens_per_s')})",
         "for r in R['grow']:",
         "    print(f\"{r['tag']:16s} val {r['val_loss']:7.4f}  MaxVio {r.get('maxvio','--')}\"",
         "          f\"  dead {r.get('dead','--')}\")",
         "for r in R['balance']:",
         "    print(f\"balance {r['balance']:5s} val {r['val_loss']:7.4f}  MaxVio {r['maxvio']}\"",
         "          f\"  dead {r['dead']}/{r['dead_of']}\")"),

    md("## What to take away",
       "",
       "- A dense model converts into a mixture of experts **without losing what it learned**, as long as",
       "  the output is scaled for the router's weights. Without that scaling the conversion throws away",
       "  about 0.8 nats on the spot.",
       "- Inherited experts beat experts started from scratch by a wide margin at this budget. The",
       "  `random` run is the same architecture with nothing carried over, and it is far behind.",
       "- Balancing does not buy loss in the short run. It buys **capacity that stays usable**: with no",
       "  balancing, experts go dead and never come back.",
       "- On one GPU the mixture of experts is slower at equal active parameters, because 32 small matrix",
       "  multiplications per layer replace one large one. That is a dispatch artifact, not the",
       "  communication cost a real multi-GPU run pays — which this setup cannot show at all."),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "moe.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out.name} -- {len(CELLS)} cells")
