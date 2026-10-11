"""The four published spike detectors, scored against a trace whose spikes we planted.

Session 15 lists four rules and does not say which to use. A rule is only as good as its
false-alarm rate, and none of the sources reports both numbers on the same trace, so we
build one trace, plant the spikes, and score all four.

    NaN check            Megatron-LM / PyTorch   stop at once on NaN or inf
    spike score          OLMo 2                  >= 7 sigma over the last 1,000 values
    EMA z-score          ZClip, 2025             EMA alpha 0.97, z > 2.5
    ratio to the maximum Megatron-LM             > 10x the largest value seen so far

The trace imitates a real grad-norm curve: a warmup, a decaying level, multiplicative
noise whose size falls with the level, and planted spikes of known height. A fixed
threshold is scored too, because the session says a fixed limit cannot work for a whole
run and that is a claim worth a number.
"""
import numpy as np

STEPS = 300_000
SEED = 15


def trace(steps=STEPS, seed=SEED):
    """A grad-norm trace with planted spikes. Returns (values, spike_steps, heights)."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)

    # Level: a 2,000-step warmup to 1.4, then a slow decay toward 0.25.
    warm = np.clip(t / 2_000, 0, 1)
    level = warm * (0.25 + 1.15 * np.exp(-t / 90_000))
    level[:50] = np.linspace(0.02, level[50], 50)

    # Lognormal noise, so the spread scales with the level the way a real trace does.
    noise = np.exp(rng.normal(0.0, 0.11, steps))
    v = level * noise

    # Plant 40 spikes, heights 3x to 40x the local level, more of them early.
    n_spike = 40
    pos = np.sort(rng.choice(np.arange(3_000, steps - 100), n_spike, replace=False))
    heights = np.exp(rng.uniform(np.log(3), np.log(40), n_spike))
    for p, h in zip(pos, heights):
        v[p] *= h
        # A real spike decays over a few steps rather than vanishing.
        for j in range(1, 5):
            if p + j < steps:
                v[p + j] *= 1 + (h - 1) * np.exp(-j / 1.2) / h * 0.7
    return v, pos, heights


# ------------------------------------------------------------------ the four rules

def rule_fixed(v, limit=5.0):
    return np.flatnonzero(v > limit)


def rule_spike_score(v, window=1_000, sigma=7.0):
    """OLMo 2: 7 sigma from the mean of the last 1,000 values. Trailing, no leakage."""
    out = []
    s = s2 = 0.0
    buf = np.empty(window)
    n = 0
    for i, x in enumerate(v):
        if n == window:
            old = buf[i % window]
            s -= old
            s2 -= old * old
            m = s / (window - 1 + 1)
            var = max(s2 / window - (s / window) ** 2, 1e-12)
            if x - s / window > sigma * np.sqrt(var):
                out.append(i)
        buf[i % window] = x
        s += x
        s2 += x * x
        n = min(n + 1, window)
    return np.array(out, dtype=int)


def rule_ema_z(v, alpha=0.97, z=2.5, warmup=100):
    """ZClip: EMA of the mean and of the variance, z-score against them."""
    out = []
    mu = v[0]
    var = 0.0
    for i, x in enumerate(v):
        if i >= warmup:
            sd = np.sqrt(max(var, 1e-12))
            if sd > 0 and (x - mu) / sd > z:
                out.append(i)
                continue          # a detected spike does not update the statistics
        d = x - mu
        mu += (1 - alpha) * d
        var = alpha * var + (1 - alpha) * d * d
    return np.array(out, dtype=int)


def rule_ratio_max(v, factor=10.0):
    """Megatron-LM: larger than `factor` times the largest value seen before it."""
    out = []
    run_max = -np.inf
    for i, x in enumerate(v):
        if run_max > 0 and x > factor * run_max:
            out.append(i)
        run_max = max(run_max, x)
    return np.array(out, dtype=int)


def rule_nan(v):
    return np.flatnonzero(~np.isfinite(v))


# ------------------------------------------------------------------ scoring

def score(found, planted, tol=4):
    """A planted spike counts as found if any alarm lands within `tol` steps of it."""
    found = np.asarray(found)
    hit = np.zeros(len(planted), dtype=bool)
    claimed = np.zeros(len(found), dtype=bool)
    for k, p in enumerate(planted):
        near = np.flatnonzero(np.abs(found - p) <= tol)
        if len(near):
            hit[k] = True
            claimed[near] = True
    return dict(found=int(hit.sum()), missed=int((~hit).sum()),
                false_alarms=int((~claimed).sum()), alarms=int(len(found)))


def loss_trace(steps=STEPS, seed=SEED + 1):
    """A loss curve with the same planted spikes, for the rules meant for the loss.

    Megatron-LM's ratio-to-the-maximum rule is written for the loss, not the grad norm,
    and the two curves behave differently: a loss falls monotonically apart from noise,
    so the running maximum is set early and stays. We score that rule on both.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    level = 1.6 + 8.0 * np.exp(-t / 12_000) + 1.2 * np.exp(-t / 150_000)
    v = level + rng.normal(0.0, 0.035, steps) * np.sqrt(level)
    pos = np.sort(rng.choice(np.arange(3_000, steps - 100), 40, replace=False))
    heights = np.exp(rng.uniform(np.log(1.3), np.log(4.0), 40))
    for p, h in zip(pos, heights):
        v[p] *= h
        for j in range(1, 6):
            if p + j < steps:
                v[p + j] *= 1 + (h - 1) * np.exp(-j / 1.5) * 0.6
    return v, pos, heights


def per_10k(s, steps):
    return round(s["false_alarms"] * 10_000 / steps, 2)
