"""The five per-layer routing metrics, and the fault that only they can see.

Section 7 of the brief reports a fault in the reference model: in layers 0 and 1 the
balancing bias overrode the router on 98.5% of choices, against a ceiling of
1 - 6/384 = 98.4%. Average MaxVio was 0.31, so the model looked balanced. Replacing the
chosen experts with random experts, and then removing the routed experts entirely, did
not change the loss. Those two layers were not using their experts at all, and a
327.6B-token test run did not show it.

Session 14's build measured MaxVio and dead experts. Neither can see this. The metric
that can is the top-k intervention rate: the share of choices the bias changes.

    dead_fraction          experts that received no tokens
    router_entropy         how strongly the router prefers some experts, in nats
    intervention_rate      share of chosen expert SLOTS that the router would not have
                           chosen -- per slot, not per token (see below)
    set_change_rate        share of tokens whose chosen SET differs at all, for contrast
    intrinsic_topk_mass    router probability mass on the router's OWN first k choices
    selected_topk_mass     router probability mass on the experts actually used

The intervention rate has to be counted per slot, not per token. If the bias picked a
k-subset at random, independent of the router, the expected overlap with the router's own
top-k is k*k/N, so the expected per-slot intervention is exactly 1 - k/N -- which is the
ceiling the reference model's 98.5% is read against (1 - 6/384 = 98.4%). Counted per
token instead, "the set changed at all" saturates near 1 for any non-zero bias and has no
such bound, so it cannot be compared to that ceiling at all. Both are reported here
because the contrast is the point: a high set-change rate is normal, a high per-slot
intervention rate is not.

When selected mass falls to k/N the router has no preference at all for what was run, and
intervention near 1 - k/N means the bias is choosing everything. Those two together are
the signature; neither alone is enough.
"""
import sys
from pathlib import Path

import torch

S14 = Path(__file__).resolve().parent.parent / "session-14-moe"
sys.path.insert(0, str(S14))
import model as M                                             # noqa: E402


class Probe:
    """Collects the five metrics for one MoE layer over a measurement window."""

    def __init__(self, layer, n_expert, top_k):
        self.lay, self.N, self.k = layer, n_expert, top_k
        self.reset()

    def reset(self):
        self.load = torch.zeros(self.N)
        self.n = 0
        self.ent = 0.0
        self.interv = 0.0
        self.changed = 0
        self.intr_mass = 0.0
        self.sel_mass = 0.0

    def observe(self, flat):
        """`flat` is the layer's input, (tokens, d_model). Recomputes routing, no grad."""
        with torch.no_grad():
            scores = torch.softmax(self.lay.router(flat.float()), dim=-1)
            bias = getattr(self.lay, "bias", torch.zeros(self.N, device=scores.device))
            chosen = (scores + bias).topk(self.k, dim=-1).indices
            intrinsic = scores.topk(self.k, dim=-1).indices

            a = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, chosen, True)
            b = torch.zeros_like(a).scatter_(1, intrinsic, True)
            # Per slot: how many of the k experts actually run were not the router's own
            # choices. This is the quantity with a 1 - k/N ceiling.
            overlap = (a & b).sum(dim=1).float()
            self.interv += float((1.0 - overlap / self.k).sum())
            # Per token: did the set change at all. No ceiling; reported for contrast.
            self.changed += int((a != b).any(dim=1).sum())

            self.load += torch.bincount(chosen.reshape(-1).cpu(),
                                        minlength=self.N).float()
            p = scores.clamp_min(1e-9)
            self.ent += float(-(p * p.log()).sum(-1).sum())
            self.intr_mass += float(scores.gather(1, intrinsic).sum())
            self.sel_mass += float(scores.gather(1, chosen).sum())
            self.n += flat.shape[0]

    def report(self):
        n = max(self.n, 1)
        mean = self.load.mean()
        return dict(
            tokens=self.n,
            dead=int((self.load == 0).sum()),
            dead_fraction=round(float((self.load == 0).float().mean()), 4),
            maxvio=round(float((self.load.max() - mean) / mean), 4) if mean > 0 else None,
            router_entropy=round(self.ent / n, 4),
            entropy_uniform=round(float(torch.log(torch.tensor(float(self.N)))), 4),
            intervention_rate=round(self.interv / n, 4),
            intervention_ceiling=round(1 - self.k / self.N, 4),
            set_change_rate=round(self.changed / n, 4),
            intrinsic_topk_mass=round(self.intr_mass / n, 4),
            selected_topk_mass=round(self.sel_mass / n, 4),
            uniform_topk_mass=round(self.k / self.N, 6),
        )


def attach(net):
    """Hook every MoE layer so one forward pass fills every probe."""
    probes, handles = [], []
    for lay in net.moe_layers():
        p = Probe(lay, lay.router.out_features, net.cfg.top_k)
        probes.append(p)

        def hook(mod, inp, _out, p=p):
            x = inp[0]
            p.observe(x.reshape(-1, x.shape[-1]))

        handles.append(lay.register_forward_hook(hook))
    return probes, handles


# ----------------------------------------------------------------- the two ablations

class Ablate:
    """Replace a layer's routing, or its routed experts, and measure the loss change.

    These are the two tests the reference model's team ran once the five metrics pointed
    at layers 0 and 1. They are the proof, and the metrics are only the alarm:

        random   route each token to k experts chosen uniformly at random
        bypass   delete the routed experts -- the layer passes its input through

    A layer that is really using its experts loses a great deal of loss under `random`.
    A layer that is not loses nothing, and that is the finding.
    """

    def __init__(self, layer, mode, top_k):
        self.lay, self.mode, self.k = layer, mode, top_k
        self.h = self._saved = self._bias = None

    def __enter__(self):
        if self.mode == "random":
            # Replace the router's output with noise. argmax of noise is a uniformly
            # random set, so this is random routing without touching Session 14's model.
            router = self.lay.router
            self._saved = router.forward
            N = router.out_features

            def noise(x, N=N):
                return torch.randn(x.shape[0], N, device=x.device, dtype=torch.float32)

            router.forward = noise
            self._bias = self.lay.bias.detach().clone()
            self.lay.bias.zero_()
        elif self.mode == "bypass":
            # Delete the routed experts: the layer contributes nothing to the residual.
            def hook(mod, inp, out):
                return torch.zeros_like(out)

            self.h = self.lay.register_forward_hook(hook)
        else:
            raise ValueError(self.mode)
        return self

    def __exit__(self, *a):
        if self.mode == "random":
            self.lay.router.forward = self._saved
            self.lay.bias.copy_(self._bias)
        else:
            self.h.remove()
        return False


# ----------------------------------------------------------------- the experiment

def measure(net, batches, iters, top_k):
    """Fill the probes over `iters` batches and return one report per layer."""
    probes, handles = attach(net)
    net.eval()
    with torch.no_grad():
        for _ in range(iters):
            x, y = batches()
            net(x, y)
    for h in handles:
        h.remove()
    return [p.report() for p in probes]


def ablate_layers(net, batches, iters, top_k, layers=None):
    """Loss change when each layer's routing is randomised, and when it is deleted."""
    import train as T

    base = T.evaluate(net, batches, iters=iters)
    out = []
    mls = net.moe_layers()
    for i, lay in enumerate(mls):
        if layers is not None and i not in layers:
            continue
        with Ablate(lay, "random", top_k):
            r = T.evaluate(net, batches, iters=iters)
        with Ablate(lay, "bypass", top_k):
            b = T.evaluate(net, batches, iters=iters)
        out.append(dict(layer=i, base=round(base, 4),
                        random=round(r, 4), random_delta=round(r - base, 4),
                        bypass=round(b, 4), bypass_delta=round(b - base, 4)))
    return base, out
