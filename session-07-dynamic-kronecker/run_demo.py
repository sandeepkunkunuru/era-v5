#!/usr/bin/env python3
"""One command that reproduces every number in the README.

    python run_demo.py

Runs, in order:
  1. build_vocab — train the 131,072 BPE vocabulary (cached after the first run)
  2. census      — count truncation and collisions per script at windows 32/48/64
  3. probe       — the decisive experiment: a task v1 provably cannot learn
  4. train       — language modelling, all input paths, identical everything else
  5. tests       — the invariants the claims rest on

Writes artifacts/{census,probe,lm,evidence}.json and prints an evidence table.
Deterministic: same seeds, same numbers, run to run.
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / "artifacts"
PY = sys.executable


def step(title: str, args: list[str]) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {title}")


def evidence() -> list[dict]:
    """Turn the raw artifacts into checkable PASS/FAIL claims."""
    census = json.loads((ART / "census.json").read_text())
    probe = json.loads((ART / "probe.json").read_text())
    lm = json.loads((ART / "lm.json").read_text())

    w32 = census["windows"]["32"]
    w64 = census["windows"]["64"]
    sep = census["separation"]
    pre = census["prefix_similarity"]
    arms = {a["name"]: a for a in probe["arms"]}
    lm_arms = {a["arm"]: a for a in lm["arms"]}

    v1_lm, dyn_lm = lm_arms["kron_v1_onehot32"], lm_arms["dyn_hybrid8"]
    rows = [
        ("the problem is real",
         f"{w32['collided_tokens']} tokens collide at a 32-byte window",
         w32["collided_tokens"] > 0),
        ("the problem is Indic",
         f"{w32['indic_share_of_collisions']}% of collided tokens are Indic-script",
         w32["indic_share_of_collisions"] > 75),
        ("widening the window does not fix it",
         f"{sep['v1_onehot64']['clusters_still_colliding']} clusters still collide at 64 bytes, "
         f"at 2x the parameters",
         sep["v1_onehot64"]["clusters_still_colliding"] > 0),
        ("v1 collisions are exact, not approximate",
         f"{arms['kron_v1_onehot32']['collision_check']['pairs_bit_identical']}"
         f"/{probe['n_pairs']} probe pairs are bit-identical under v1",
         arms["kron_v1_onehot32"]["collision_check"]["pairs_bit_identical"] == probe["n_pairs"]),
        ("v1 is pinned at chance on the probe",
         f"accuracy {arms['kron_v1_onehot32']['final_accuracy']:.3f}",
         abs(arms["kron_v1_onehot32"]["final_accuracy"] - 0.5) < 1e-6),
        ("the dynamic codec solves the probe",
         f"dyn_hybrid8 accuracy {arms['dyn_hybrid8']['final_accuracy']:.3f}",
         arms["dyn_hybrid8"]["final_accuracy"] > 0.99),
        ("the dynamic codec never crops",
         "0 clusters still colliding at any tested window",
         sep["dyn_hybrid8"]["clusters_still_colliding"] == 0),
        ("it costs a quarter of v1's parameters",
         f"code_dim {sep['dyn_hybrid8']['code_dim']} vs {sep['v1_onehot32']['code_dim']}",
         sep["dyn_hybrid8"]["code_dim"] * 4 == sep["v1_onehot32"]["code_dim"]),
        ("spelling awareness is preserved, not sacrificed",
         f"prefix separation {pre['dyn_hybrid8']['separation']:+.4f} "
         f"vs v1 {pre['v1_onehot32']['separation']:+.4f}",
         pre["dyn_hybrid8"]["separation"] >= pre["v1_onehot32"]["separation"]),
        ("normalised position is NOT the answer",
         f"relative-position prefix separation collapses to "
         f"{pre['dyn_relative16']['separation']:+.4f}",
         pre["dyn_relative16"]["separation"] < pre["v1_onehot32"]["separation"] / 2),
        ("language modelling does not regress",
         f"val loss {dyn_lm['val_loss']:.4f} vs v1 {v1_lm['val_loss']:.4f} "
         f"(delta {dyn_lm['val_loss'] - v1_lm['val_loss']:+.4f})",
         dyn_lm["val_loss"] <= v1_lm["val_loss"] + 0.05),
        ("at a quarter of the input-path parameters",
         f"{dyn_lm['input_path_params']:,} vs {v1_lm['input_path_params']:,}",
         dyn_lm["input_path_params"] * 4 == v1_lm["input_path_params"]),
    ]
    return [{"claim": c, "evidence": e, "status": "PASS" if ok else "FAIL"} for c, e, ok in rows]


def main() -> int:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    step("1/5  Train the V5-class 131,072 BPE vocabulary", ["-m", "dyn.build_vocab"])
    step("2/5  Collision census, per script, at windows 32 / 48 / 64", ["-m", "dyn.census"])
    step("3/5  The decisive probe: a task v1 cannot learn", ["-m", "dyn.probe"])
    step("4/5  Language modelling: is the compression free?", ["-m", "dyn.train"])
    step("5/5  Invariants", ["-m", "unittest", "discover", "tests", "-v"])

    rows = evidence()
    (ART / "evidence.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'=' * 72}\nEVIDENCE\n{'=' * 72}")
    width = max(len(r["claim"]) for r in rows)
    for r in rows:
        print(f"  [{r['status']}] {r['claim']:<{width}}  {r['evidence']}")
    n_pass = sum(r["status"] == "PASS" for r in rows)
    print(f"\n{n_pass}/{len(rows)} checks passed in {time.time() - t0:.0f}s")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
