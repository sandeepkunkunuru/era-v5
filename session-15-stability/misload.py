"""The silent checkpoint misload, reproduced, with the five cheap checks passing.

The LightningLM cookbook (lightninglm.theschoolofai.in) calls this "the most dangerous
failure in the project, because for a period it did not present as a failure". Its §7.2
lists five checks that all passed, and reports that the only signal was the loss at step
one: about 2.95 against about 2.55 for a correct load.

We reproduce it on Session 14's model. A converter writes the tensors under source-style
key names; the target model wraps its stack differently, so the names do not match; a
non-strict `load_state_dict` skips them and reports success, and those parameters keep
their random initialization.

Then we run the five checks and show each one passing, and show the two checks that
actually catch it: a strict load, and a per-parameter comparison against the source.

    python misload.py
"""
import hashlib
import sys
from pathlib import Path

import torch

S14 = Path(__file__).resolve().parent.parent / "session-14-moe"
sys.path.insert(0, str(S14))
import model as M                                             # noqa: E402

# The five checks the cookbook lists, in its order.
CHECKS = [
    "checkpoint file exists",
    "checkpoint is the right size",
    "the model load returns without error",
    "parameter count looks right",
    "training starts and the loss is finite",
]


class Wrapped(torch.nn.Module):
    """The target model, wrapped the way a sharded or compiled stack wraps it.

    This is the whole mechanism: the parameters are the same tensors, but every key
    now begins with `inner.`, so a checkpoint written under the bare names matches
    nothing and `strict=False` silently keeps the random initialization.
    """

    def __init__(self, cfg):
        super().__init__()
        self.inner = M.GPT(cfg)

    def forward(self, *a, **k):
        return self.inner(*a, **k)


def tensor_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def source_style(sd, frac=1.0, seed=0):
    """What the broken converter wrote: the source model's own key names.

    `frac` is the share of tensors written under the WRONG namespace. The cookbook's
    converter misnamed one class of tensor rather than all of them, and that is what
    makes the failure dangerous: a partial misload leaves most of the model correct, so
    the step-one loss rises by a fraction of a nat instead of by several, and a fraction
    of a nat reads as initialization noise.
    """
    import random
    keys = sorted(sd)
    r = random.Random(seed)
    wrong = set(r.sample(keys, int(round(frac * len(keys)))))
    out = {}
    for k in keys:
        out[k if k in wrong else f"inner.{k}"] = sd[k].clone()
    return out


def target_keyspace(target, sd):
    """The correction from §5.4: walk the TARGET's parameters and find each one's source.

    The direction of the transfer is inverted. Every target name must be satisfied, so a
    missing or misnamed parameter becomes a construction-time error, not a silent random
    fallback.
    """
    out = {}
    missing = []
    for name in target.state_dict():
        src = name.split("inner.", 1)[1] if name.startswith("inner.") else name
        if src in sd:
            out[name] = sd[src].clone()
        else:
            missing.append(name)
    if missing:
        raise KeyError(f"{len(missing)} target parameters have no source: {missing[:3]}")
    return out


def step_one_loss(net, batches, iters=12):
    """The loss at step one -- the one number that showed the fault."""
    net.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = batches()
            tot += net(x, y)[1].item()
    return tot / iters


def main():
    import json

    import train as T                                          # noqa: E402

    dense_pt = S14 / "dense.pt"
    if not dense_pt.exists():
        sys.exit(f"missing {dense_pt} -- run Session 14's train.py first")

    cfg = M.Config()
    source = torch.load(dense_pt, map_location="cpu")
    torch.manual_seed(15)
    good = Wrapped(cfg)
    torch.manual_seed(16)
    bad = Wrapped(cfg)

    ckpt = Path("/tmp/s15_misload.pt")
    written = source_style(source, frac=1.0)
    torch.save(written, ckpt)

    # --- the broken load: source-style names into a wrapped target, non-strict
    before = tensor_hash(bad.state_dict())
    result = bad.load_state_dict(written, strict=False)
    after = tensor_hash(bad.state_dict())

    # --- the corrected load: target keyspace first, then strict
    good.load_state_dict(target_keyspace(good, source), strict=True)

    batches = T.Batches("val", 8)
    T.DEV = "cpu"
    bad_loss = step_one_loss(bad, batches)
    good_loss = step_one_loss(good, batches)

    n_bad = sum(p.numel() for p in bad.parameters())
    n_good = sum(p.numel() for p in good.parameters())
    passed = [
        (CHECKS[0], ckpt.exists()),
        (CHECKS[1], ckpt.stat().st_size > 50_000_000),
        (CHECKS[2], True),                       # strict=False returned, no exception
        (CHECKS[3], n_bad == n_good),
        (CHECKS[4], bool(torch.isfinite(torch.tensor(bad_loss)))),
    ]

    print("the five cheap checks, on the BROKEN load")
    for name, ok in passed:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  missing_keys    {len(result.missing_keys):,} "
          f"(every parameter of the target, kept at its random initialization)")
    print(f"  unexpected_keys {len(result.unexpected_keys):,} "
          f"(every tensor in the checkpoint, written and then ignored)")
    print(f"  state hash before load {before} · after load {after} · "
          f"{'UNCHANGED' if before == after else 'changed'}")

    print(f"\nstep-one loss  broken {bad_loss:.4f}  ·  correct {good_loss:.4f}  "
          f"·  gap {bad_loss - good_loss:+.4f} nats")

    # --- the two checks that do catch it
    print("\nthe checks that catch it")
    try:
        Wrapped(cfg).load_state_dict(written, strict=True)
        strict_caught = False
    except RuntimeError as e:
        strict_caught = True
        print(f"  strict load         raised: {str(e).splitlines()[0][:76]}")
    tgt_caught = False
    try:
        target_keyspace(Wrapped(cfg), {k: v for k, v in source.items() if "wte" not in k})
    except KeyError as e:
        tgt_caught = True
        print(f"  target-keyspace     raised: {str(e)[:76]}")

    # --- how small can the only signal get? sweep the share of misnamed tensors
    print("\nthe signal, against the share of tensors misnamed")
    print("  misnamed   params random      step-one loss    gap")
    sweep = []
    for frac in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        torch.manual_seed(16)
        net = Wrapped(cfg)
        sd = source_style(source, frac=frac, seed=3)
        res = net.load_state_dict(sd, strict=False)
        rand = sum(p.numel() for n, p in net.named_parameters() if n in set(res.missing_keys))
        L = step_one_loss(net, batches)
        sweep.append(dict(frac=frac, tensors_misnamed=len(res.missing_keys),
                          params_random=rand, loss=round(L, 4),
                          gap=round(L - good_loss, 4)))
        print(f"  {frac:>6.0%}   {rand:>12,}      {L:>8.4f}      {L - good_loss:+.4f}")

    # Step-to-step noise on a correct model, for comparison: a gap smaller than this
    # cannot be seen at all without a reference run.
    noise = []
    for _ in range(12):
        noise.append(step_one_loss(good, batches, iters=1))
    import statistics
    sd_noise = statistics.pstdev(noise)
    print(f"\n  batch-to-batch spread of the correct model: {sd_noise:.4f} nats (1 sigma)")
    print(f"  the smallest gap above 2 sigma: "
          f"{min((s['gap'] for s in sweep if s['gap'] > 2 * sd_noise), default=float('nan')):.4f}")

    out = dict(
        sweep=sweep, batch_noise_sigma=round(sd_noise, 4),
        broken_loss=round(bad_loss, 4), correct_loss=round(good_loss, 4),
        gap=round(bad_loss - good_loss, 4),
        missing_keys=len(result.missing_keys), unexpected_keys=len(result.unexpected_keys),
        state_unchanged_by_load=before == after,
        params_broken=n_bad, params_correct=n_good,
        cheap_checks={n: bool(v) for n, v in passed},
        strict_load_catches=strict_caught, target_keyspace_catches=tgt_caught,
        cookbook_reference=dict(broken=2.95, correct=2.55, gap=0.40,
                                source="LightningLM cookbook §7.2"),
    )
    Path("misload.json").write_text(json.dumps(out, indent=2))
    ckpt.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    main()
