"""Freezing: the four constraints that break silently, each one made to break.

Sections 11 and 12 of the brief list rules the training code must obey when part of the
model is frozen, and say that ignoring them usually produces no error. We reproduce each
failure on Session 14's model, measure it, and then show the check that catches it.

    1. optimizer groups   a weight frozen AFTER the optimizer was built still decays
    2. frozen hash        a hash at every checkpoint is what proves it did not move
    3. zero-grad counts   per part, the number the dashboard watches
    4. the grad norm      freezing changes its size, so the clip limit no longer means
                          what it meant
    5. memory             a frozen weight keeps 2 of its 16 bytes

    python freeze.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

S14 = Path(__file__).resolve().parent.parent / "session-14-moe"
sys.path.insert(0, str(S14))
import model as M                                             # noqa: E402

BYTES_TRAINED = 16   # 2 weight + 2 master grad + 4 fp32 master + 4 exp_avg + 4 exp_avg_sq
BYTES_FROZEN = 2


def whash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def parts(net):
    """The model split the way a dashboard splits it, for per-part zero-grad counts."""
    out = {"embedding": [], "attention": [], "experts": [], "router": [], "norms": [],
           "head": []}
    for n, p in net.named_parameters():
        if n.startswith(("wte", "wpe")):
            out["embedding"].append((n, p))
        elif ".attn." in n:
            out["attention"].append((n, p))
        elif ".router" in n or n.endswith(".bias") and ".mlp" in n:
            out["router"].append((n, p))
        elif ".mlp." in n:
            out["experts"].append((n, p))
        elif ".ln" in n or n.startswith("ln_f"):
            out["norms"].append((n, p))
        else:
            out["head"].append((n, p))
    return {k: v for k, v in out.items() if v}


def zero_grad_counts(net):
    """Per part: parameters with no gradient at all, and elements that are exactly zero."""
    rep = {}
    for name, ps in parts(net).items():
        no_grad = sum(1 for _, p in ps if p.requires_grad and p.grad is None)
        exact_zero = sum(int((p.grad == 0).sum()) for _, p in ps if p.grad is not None)
        total = sum(p.numel() for _, p in ps)
        rep[name] = dict(tensors=len(ps), params=total, trainable_without_grad=no_grad,
                         zero_grad_elements=exact_zero,
                         zero_grad_share=round(exact_zero / max(total, 1), 4))
    return rep


def grad_norm(net, trainable_only=True):
    g = [p.grad for p in net.parameters()
         if p.grad is not None and (p.requires_grad or not trainable_only)]
    if not g:
        return 0.0
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g)))


def memory_bytes(net):
    tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    return dict(trainable_params=tr, frozen_params=fr,
                bytes_trained=tr * BYTES_TRAINED, bytes_frozen=fr * BYTES_FROZEN,
                total_bytes=tr * BYTES_TRAINED + fr * BYTES_FROZEN,
                all_trainable_bytes=(tr + fr) * BYTES_TRAINED)


def main():
    import train as T                                          # noqa: E402

    torch.manual_seed(15)
    cfg = M.Config(moe=True, n_expert=32, top_k=8, balance="bias")
    net = M.GPT(cfg)
    batches = T.Batches("train", 4)
    T.DEV = "cpu"
    out = {}

    # ---------------------------------------------------------------- 1. optimizer groups
    # The wrong order: build the optimizer over every parameter, THEN freeze. AdamW's
    # decoupled weight decay does not need a gradient, so it keeps shrinking the frozen
    # weight at every step.
    LR, WD = 1e-3, 0.1
    bad = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    frozen = [p for n, p in net.named_parameters() if n.startswith("wte")]
    for p in frozen:
        # The stale gradient is the mechanism. PyTorch skips a parameter whose grad is
        # None, so a weight frozen with `requires_grad_(False)` alone is safe -- until a
        # zero gradient from before the freeze is still attached, which is exactly the
        # case the brief describes. Then AdamW processes it and decoupled weight decay
        # shrinks it by lr * wd every step, with no gradient involved at all.
        p.grad = torch.zeros_like(p)
        p.requires_grad_(False)
    h0 = whash(frozen[0])
    v0 = frozen[0].detach().clone()
    for _ in range(20):
        bad.step()
    drift = float((frozen[0] - v0).abs().max())
    out["decay_on_frozen"] = dict(
        steps=20, lr=0.0, weight_decay=0.1,
        max_abs_drift=round(drift, 8),
        relative=round(drift / float(v0.abs().max()), 6),
        hash_before=h0, hash_after=whash(frozen[0]),
        hash_catches=h0 != whash(frozen[0]))
    expected = 1 - (1 - LR * WD) ** 20
    out["decay_on_frozen"].update(lr=LR, weight_decay=WD,
                                  predicted_shrink=round(expected, 6))
    print(f"1. optimizer built before freezing, then a stale zero gradient, 20 steps")
    print(f"   predicted shrink 1-(1-lr*wd)^20 = {expected:.4%}")
    print(f"   frozen weight moved by {drift:.3e} "
          f"({out['decay_on_frozen']['relative']:.4%} of its largest element)")
    print(f"   hash check: {'CAUGHT' if out['decay_on_frozen']['hash_catches'] else 'missed'}")

    # The right order: build the optimizer from trainable parameters only.
    frozen[0].copy_(v0)
    for p in frozen:
        p.grad = None
    good = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                             lr=LR, weight_decay=WD)
    h1 = whash(frozen[0])
    for _ in range(20):
        good.step()
    out["decay_on_frozen"]["fixed_drift"] = round(float((frozen[0] - v0).abs().max()), 10)
    out["decay_on_frozen"]["fixed_hash_stable"] = h1 == whash(frozen[0])
    print(f"   optimizer from trainable parameters only: drift "
          f"{out['decay_on_frozen']['fixed_drift']:.1e}, hash stable "
          f"{out['decay_on_frozen']['fixed_hash_stable']}")

    # ---------------------------------------------------------------- 2/3. grads, counts
    for p in net.parameters():
        p.requires_grad_(True)
    net.zero_grad(set_to_none=True)
    x, y = batches()
    net(x, y)[1].backward()
    full_norm = grad_norm(net)
    out["zero_grad_all_trainable"] = zero_grad_counts(net)

    # Freeze everything but the experts and the router, the shape the brief describes.
    for n, p in net.named_parameters():
        p.requires_grad_(".mlp." in n)
    net.zero_grad(set_to_none=True)
    net(x, y)[1].backward()
    frozen_norm = grad_norm(net)
    out["zero_grad_experts_only"] = zero_grad_counts(net)

    out["grad_norm"] = dict(
        all_trainable=round(full_norm, 4), experts_only=round(frozen_norm, 4),
        ratio=round(frozen_norm / full_norm, 4),
        note="the clip limit and the alarm band are set on this number, so freezing moves both")
    print(f"\n2. grad norm over all weights {full_norm:.4f} · over the trainable ones "
          f"{frozen_norm:.4f} · ratio {frozen_norm / full_norm:.3f}")
    print(f"   a clip limit of 1.0 now bites at a different place")

    tr = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    print(f"\n3. trainable parameters that received NO gradient: {len(tr)}")
    for name, rep in out["zero_grad_experts_only"].items():
        print(f"   {name:10s} {rep['params']:>11,} params · "
              f"{rep['zero_grad_share']:>7.2%} of gradient elements exactly zero")

    # ---------------------------------------------------------------- 4. memory
    out["memory"] = memory_bytes(net)
    m = out["memory"]
    # The other shape: freeze the experts, which is where the parameters are, and train
    # everything else. This is the Lightning LM / QLoRA arrangement.
    flags = {n: p.requires_grad for n, p in net.named_parameters()}
    for n, p in net.named_parameters():
        p.requires_grad_(".mlp." not in n or ".router" in n)
    out["memory_experts_frozen"] = memory_bytes(net)
    for n, p in net.named_parameters():
        p.requires_grad_(flags[n])
    print(f"\n4. memory, at 16 bytes a trained weight and 2 a frozen one")
    print(f"   trainable {m['trainable_params']:>12,} · frozen {m['frozen_params']:>12,}")
    print(f"   experts trained, rest frozen: {m['total_bytes'] / 2**20:,.0f} MiB "
          f"against {m['all_trainable_bytes'] / 2**20:,.0f} MiB with nothing frozen "
          f"({m['all_trainable_bytes'] / m['total_bytes']:.2f}x)")
    e = out["memory_experts_frozen"]
    print(f"   experts FROZEN, rest trained: {e['total_bytes'] / 2**20:,.0f} MiB "
          f"({e['all_trainable_bytes'] / e['total_bytes']:.2f}x) -- the experts hold "
          f"{m['trainable_params'] / (m['trainable_params'] + m['frozen_params']):.0%} "
          f"of the parameters, so which side is frozen decides everything")

    # ---------------------------------------------------------------- 5. frozen hashes
    hashes = {n: whash(p) for n, p in net.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(5):
        net.zero_grad(set_to_none=True)
        xb, yb = batches()
        net(xb, yb)[1].backward()
        opt.step()
    moved = [n for n, p in net.named_parameters()
             if not p.requires_grad and whash(p) != hashes[n]]
    out["frozen_hash_after_5_steps"] = dict(checked=len(hashes), moved=len(moved),
                                            moved_names=moved[:5])
    print(f"\n5. after 5 real steps: {len(hashes)} frozen tensors hashed, "
          f"{len(moved)} changed")

    Path("freeze.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
