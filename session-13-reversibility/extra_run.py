"""A fourth run, to remove a confound in run 3.

Run 3 trains the same 50M tokens at a much larger batch, so it takes far fewer optimizer steps —
638 against 3,756 — at the same learning rate. That is two changes at once, and the loss difference
cannot be attributed to the batch size alone.

Session 11's rule for Adam is that the learning rate should rise roughly with the square root of the
batch ratio. This run repeats run 3 with exactly that correction and nothing else, so run 3 and run 4
differ only in the learning rate.

    python extra_run.py        # appends run 4 to results.json
"""
import json
import math
from pathlib import Path

import train as T

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
runs = {r["tag"]: r for r in R["runs"]}
base, big = runs["1-baseline"], runs["3-reversible-max-batch"]

ratio = big["batch"] / base["batch"]
scaled = T.LR * math.sqrt(ratio)
print(f"batch {base['batch']} -> {big['batch']} ({ratio:.1f}x): learning rate "
      f"{T.LR:.2e} -> {scaled:.2e} (square-root rule, Session 11)", flush=True)

T.LR = scaled
row = T.run(big["stack"], big["batch"], R["tokens"], "4-reversible-max-batch-lr-scaled")
row["lr"] = scaled
row["note"] = (f"run 3 repeated with the learning rate scaled by sqrt({ratio:.1f}) = "
               f"{math.sqrt(ratio):.2f}, so it differs from run 3 only in the learning rate")

R["runs"] = [r for r in R["runs"] if r["tag"] != row["tag"]] + [row]
R["lr_scaled"] = dict(base_lr=3e-4, scaled_lr=scaled, batch_ratio=ratio)
(HERE / "results.json").write_text(json.dumps(R, indent=1))
print("appended run 4 to results.json", flush=True)
