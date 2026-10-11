"""Downsampling a 300,000-step curve onto a 2,000-pixel screen, and what each method loses.

Section 16 of the brief gives the arithmetic -- about 150 steps per pixel column -- and the
claim that a mean-only rollup hides a spike. It does not give the error, so we measure it.

Methods:
    mean        one value per column: the mean
    minmax      two values per column: the minimum and the maximum
    M4          four values per column: the min, the max, the first and the last
    lttb        Largest-Triangle-Three-Buckets, the usual alternative, for contrast

M4 (Jugel et al., EDBT 2014) is the one with a proof attached: for a line chart rasterised
at a given width, drawing min/max/first/last per column reproduces the full-data rendering
pixel for pixel. We check that claim directly by rasterising both and comparing pixels.
"""
import numpy as np


def columns(n, width):
    """The index range of each pixel column.

    The bucket boundaries must be the same mapping the rasteriser uses to place a point,
    x = round(i / (n - 1) * (width - 1)), or M4's buckets straddle two pixel columns and
    the proof no longer applies. Getting this wrong costs about 2,700 wrong pixels and
    looks like a flaw in M4 rather than a flaw in the bucketing.
    """
    i = np.arange(n)
    col = np.round(i / max(n - 1, 1) * (width - 1)).astype(int)
    edges = np.flatnonzero(np.diff(col, prepend=-1))
    return list(zip(edges, list(edges[1:]) + [n]))


def ds_mean(y, width):
    out = []
    for a, b in columns(len(y), width):
        if b > a:
            out.append(((a + b - 1) / 2, y[a:b].mean()))
    return np.array(out)


def ds_minmax(y, width, per_column=False):
    out = []
    for a, b in columns(len(y), width if per_column else width // 2):
        if b <= a:
            continue
        seg = y[a:b]
        i, j = a + int(seg.argmin()), a + int(seg.argmax())
        for k in sorted((i, j)):
            out.append((k, y[k]))
    return np.array(out)


def ds_m4(y, width, per_column=False):
    """`per_column=True` is M4 as the paper states it: one bucket per pixel column,
    up to four points each. That is the form the no-visible-error proof applies to.
    `per_column=False` squeezes M4 into the same point budget as the other methods,
    which is the fairer comparison but is not what the proof covers."""
    out = []
    for a, b in columns(len(y), width if per_column else width // 4):
        if b <= a:
            continue
        seg = y[a:b]
        idx = {a, b - 1, a + int(seg.argmin()), a + int(seg.argmax())}
        for k in sorted(idx):
            out.append((k, y[k]))
    return np.array(out)


def ds_lttb(y, width):
    n = len(y)
    if width >= n:
        return np.stack([np.arange(n), y], 1)
    buckets = columns(n - 2, width - 2)
    out = [(0, y[0])]
    prev = 0
    for bi, (a, b) in enumerate(buckets):
        a, b = a + 1, b + 1
        na, nb = (buckets[bi + 1][0] + 1, buckets[bi + 1][1] + 1) if bi + 1 < len(buckets) else (n - 1, n)
        ax, ay = prev, y[prev]
        nx, ny = (na + nb - 1) / 2, y[na:nb].mean()
        seg = np.arange(a, b)
        area = np.abs((ax - nx) * (y[seg] - ay) - (ax - seg) * (ny - ay))
        k = int(seg[area.argmax()])
        out.append((k, y[k]))
        prev = k
    out.append((n - 1, y[n - 1]))
    return np.array(out)


def raster(points, n, width, height, lo, hi):
    """Rasterise a polyline onto a width x height boolean grid, the way a chart does."""
    g = np.zeros((height, width), dtype=bool)
    xs = np.clip((points[:, 0] / max(n - 1, 1) * (width - 1)).round().astype(int), 0, width - 1)
    ys = np.clip(((points[:, 1] - lo) / (hi - lo) * (height - 1)).round().astype(int), 0, height - 1)
    for i in range(len(xs) - 1):
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if x1 == x0:
            g[min(y0, y1):max(y0, y1) + 1, x0] = True
            continue
        for x in range(x0, x1 + 1):
            t = (x - x0) / (x1 - x0)
            ya = int(round(y0 + t * (y1 - y0)))
            tn = (x + 1 - x0) / (x1 - x0)
            yb = int(round(y0 + min(tn, 1.0) * (y1 - y0)))
            g[min(ya, yb):max(ya, yb) + 1, x] = True
    return g
