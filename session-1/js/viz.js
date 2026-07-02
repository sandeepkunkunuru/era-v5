// Shared canvas visualization helpers: the diverging teal<->coral probability
// field, scatter points, and small line/bar charts. tf is loaded globally.

export const COLORS = {
  class0: [45, 212, 191], // teal   #2DD4BF
  class1: [251, 113, 133], // coral  #FB7185
  ink: '#0B1020',
  grid: 'rgba(230,234,242,0.08)',
  boundary: 'rgba(230,234,242,0.85)',
  text: '#8A93A8',
};

// diverging colormap: p=0 -> teal, p=0.5 -> ink, p=1 -> coral
function fieldColor(p) {
  const [t0, t1, t2] = COLORS.class0;
  const [c0, c1, c2] = COLORS.class1;
  // fade toward dark ink near the boundary so the 0.5 contour reads as a valley
  const d = Math.abs(p - 0.5) * 2; // 0 at boundary, 1 at extremes
  const strength = 0.14 + 0.72 * Math.pow(d, 0.9);
  let r, g, b;
  if (p < 0.5) {
    r = t0; g = t1; b = t2;
  } else {
    r = c0; g = c1; b = c2;
  }
  return `rgba(${Math.round(r)},${Math.round(g)},${Math.round(b)},${strength})`;
}

// Map data coords -> pixel coords for a square canvas over `bounds`.
export function makeTransform(canvas, bounds) {
  const w = canvas.width, h = canvas.height;
  const { xmin, xmax, ymin, ymax } = bounds;
  return {
    toPx: (x, y) => [
      ((x - xmin) / (xmax - xmin)) * w,
      h - ((y - ymin) / (ymax - ymin)) * h,
    ],
    w, h, bounds,
  };
}

// Render a model's probability field over the plane. `predictFn` maps an
// [N,2] tf tensor -> [N] probabilities. res = grid resolution.
export function drawField(canvas, predictFn, bounds, res = 64) {
  const ctx = canvas.getContext('2d');
  const { xmin, xmax, ymin, ymax } = bounds;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pts = [];
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      const x = xmin + ((i + 0.5) / res) * (xmax - xmin);
      const y = ymin + ((j + 0.5) / res) * (ymax - ymin);
      pts.push([x, y]);
    }
  }
  let probs;
  tf.tidy(() => {
    const t = tf.tensor2d(pts);
    probs = predictFn(t).dataSync();
  });

  const cellW = canvas.width / res;
  const cellH = canvas.height / res;
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      const p = probs[j * res + i];
      ctx.fillStyle = fieldColor(p);
      // canvas y is flipped relative to data y
      const px = i * cellW;
      const py = canvas.height - (j + 1) * cellH;
      ctx.fillRect(Math.floor(px), Math.floor(py), Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }
  }
}

// Overlay the scatter points.
export function drawPoints(canvas, X, y, bounds, opts = {}) {
  const ctx = canvas.getContext('2d');
  const tf2 = makeTransform(canvas, bounds);
  const r = opts.radius || 3.2;
  for (let i = 0; i < X.length; i++) {
    const [px, py] = tf2.toPx(X[i][0], X[i][1]);
    const col = y[i] === 0 ? COLORS.class0 : COLORS.class1;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, 2 * Math.PI);
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(11,16,32,0.9)';
    ctx.stroke();
  }
}

// Clear + faint frame.
export function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = COLORS.ink;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Simple grouped-bar chart for the data-scaling proof.
// series = [{label, values:[...], color}], categories = [labels]
export function drawBars(canvas, categories, series, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const padL = 44, padR = 12, padT = 16, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxV = opts.max || 1;

  // y gridlines
  ctx.strokeStyle = 'rgba(230,234,242,0.10)';
  ctx.fillStyle = COLORS.text;
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.textAlign = 'right';
  for (let g = 0; g <= 5; g++) {
    const v = (maxV * g) / 5;
    const yy = padT + plotH - (v / maxV) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(W - padR, yy);
    ctx.stroke();
    ctx.fillText(v.toFixed(1), padL - 6, yy + 3);
  }

  const groupW = plotW / categories.length;
  const barW = (groupW * 0.62) / series.length;
  categories.forEach((cat, ci) => {
    const gx = padL + ci * groupW + groupW * 0.19;
    series.forEach((s, si) => {
      const v = s.values[ci];
      const bh = (v / maxV) * plotH;
      const bx = gx + si * barW;
      ctx.fillStyle = s.color;
      ctx.fillRect(bx, padT + plotH - bh, barW - 3, bh);
    });
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'center';
    ctx.fillText(cat, padL + ci * groupW + groupW / 2, H - 12);
  });
}

// Scatter for arbitrary 2D points with category colors + labels (embeddings).
export function drawEmbedding(canvas, points, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const pad = 42;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  const padFrac = 0.12;
  const dx = (xmax - xmin) || 1, dy = (ymax - ymin) || 1;
  xmin -= dx * padFrac; xmax += dx * padFrac;
  ymin -= dy * padFrac; ymax += dy * padFrac;
  const toPx = (x, y) => [
    pad + ((x - xmin) / (xmax - xmin)) * (W - 2 * pad),
    H - pad - ((y - ymin) / (ymax - ymin)) * (H - 2 * pad),
  ];
  const catColors = opts.catColors || {};
  // Same-category tokens can collapse to nearly the same point (that IS the
  // proof), so nudge dots apart deterministically and stack labels so every
  // token stays readable.
  const placed = points.map((p, i) => {
    let [px, py] = toPx(p.x, p.y);
    px += ((i * 7) % 5) - 2;
    py += ((i * 13) % 5) - 2;
    return { p, px, py, lx: px + 9, ly: py + 3 };
  });
  // greedy vertical de-collision of labels
  const boxes = [];
  for (const it of placed.sort((a, b) => a.ly - b.ly)) {
    let ly = it.ly;
    let guard = 0;
    while (boxes.some((b) => Math.abs(b.lx - it.lx) < 66 && Math.abs(b.ly - ly) < 13) && guard++ < 40) ly += 13;
    it.ly = ly;
    boxes.push({ lx: it.lx, ly });
  }
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.textAlign = 'left';
  for (const it of placed) {
    const col = catColors[it.p.cat] || '#8A93A8';
    // connector when the label was pushed away from its dot
    if (Math.abs(it.ly - (it.py + 3)) > 7) {
      ctx.strokeStyle = 'rgba(230,234,242,0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(it.px + 4, it.py);
      ctx.lineTo(it.lx - 2, it.ly - 3);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(it.px, it.py, 5, 0, 2 * Math.PI);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.fillStyle = '#E6EAF2';
    ctx.fillText(it.p.label, it.lx, it.ly);
  }
}
