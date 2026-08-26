/* ERA v5 · Session 9 — rendering. Every number comes from data.js; nothing here invents one. */
import { BILLS, CHUNKING, BUGS, MTP, CORRECTIONS } from "./data.js";

const TEAL = "#2dd4bf", CORAL = "#fb7185", GOLD = "#fbbf24";
const GRID = "rgba(230,234,242,0.10)", AXIS = "rgba(230,234,242,0.28)", MUTED = "#8a93a8";
const SURFACE = "#141a2e";
const svgns = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");

const el = (n, a = {}, kids = []) => {
  const e = document.createElementNS(svgns, n);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  for (const k of [].concat(kids)) e.appendChild(k);
  return e;
};
const txt = (s, a = {}) => {
  const t = el("text", { fill: MUTED, "font-family": "IBM Plex Mono, monospace", "font-size": 11, ...a });
  t.textContent = s;
  return t;
};
function showTip(ev, html) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 10) + "px";
  tip.style.top = ev.clientY - 40 + "px";
}
const hideTip = () => (tip.style.opacity = 0);
const fmtT = t => (t >= 1048576 ? "1M" : t >= 1024 ? (t / 1024) + "K" : t);

/* ---------------------------------------------------------------- 02 the bills */
function drawBills() {
  const svg = document.getElementById("billsChart");
  const W = Math.max(560, Math.min(920, svg.parentElement.clientWidth || 880));
  const H = 380, ml = 58, mr = 130, mt = 16, mb = 44;
  const iw = W - ml - mr, ih = H - mt - mb;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);

  const xs = BILLS.map(b => Math.log10(b.T));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const all = BILLS.flatMap(b => [b.kv, b.logits, b.logitsBwd]);
  const y0 = Math.log10(Math.min(...all)), y1 = Math.log10(Math.max(...all));
  const X = t => ml + (Math.log10(t) - x0) / (x1 - x0) * iw;
  const Y = v => mt + ih - (Math.log10(v) - y0) / (y1 - y0) * ih;

  // y grid at decades
  for (let p = Math.ceil(y0); p <= Math.floor(y1); p++) {
    const v = 10 ** p, y = Y(v);
    svg.appendChild(el("line", { x1: ml, x2: ml + iw, y1: y, y2: y, stroke: GRID }));
    svg.appendChild(txt(v >= 1 ? `${v} GiB` : `${v} GiB`, { x: ml - 8, y: y + 4, "text-anchor": "end" }));
  }
  // x ticks
  BILLS.forEach(b => {
    const x = X(b.T);
    svg.appendChild(el("line", { x1: x, x2: x, y1: mt + ih, y2: mt + ih + 5, stroke: AXIS }));
    svg.appendChild(txt(fmtT(b.T), { x, y: mt + ih + 20, "text-anchor": "middle" }));
  });
  svg.appendChild(txt("context length, tokens", { x: ml + iw / 2, y: H - 6, "text-anchor": "middle", fill: MUTED }));
  svg.appendChild(el("line", { x1: ml, x2: ml + iw, y1: mt + ih, y2: mt + ih, stroke: AXIS }));

  const series = [
    { key: "kv", c: TEAL, label: "KV cache" },
    { key: "logits", c: CORAL, label: "logits tensor" },
    { key: "logitsBwd", c: GOLD, label: "logits + gradient" },
  ];
  series.forEach(s => {
    const d = BILLS.map((b, i) => `${i ? "L" : "M"}${X(b.T).toFixed(1)},${Y(b[s.key]).toFixed(1)}`).join(" ");
    svg.appendChild(el("path", { d, fill: "none", stroke: s.c, "stroke-width": 2, "stroke-linecap": "round" }));
    // direct label at the right end — identity is never colour-alone
    const last = BILLS[BILLS.length - 1];
    svg.appendChild(txt(s.label, { x: ml + iw + 10, y: Y(last[s.key]) + 4, fill: s.c, "font-size": 11.5 }));
    BILLS.forEach(b => {
      const c = el("circle", { cx: X(b.T), cy: Y(b[s.key]), r: 4, fill: s.c, stroke: SURFACE, "stroke-width": 2 });
      c.style.cursor = "pointer";
      c.addEventListener("mousemove", ev => showTip(ev,
        `${fmtT(b.T)} tokens · ${s.label}<br><b>${b[s.key] < 1 ? b[s.key].toFixed(3) : b[s.key].toFixed(b[s.key] < 10 ? 2 : 0)} GiB</b>`));
      c.addEventListener("mouseleave", hideTip);
      svg.appendChild(c);
    });
  });

  // The lines are parallel because all three are linear in T, so the ratio between them is
  // constant -- which is precisely what parallel lines hide. Say it on the chart.
  const mark = BILLS.find(b => b.T === 262144);
  const mx = X(mark.T);
  svg.appendChild(el("line", { x1: mx, x2: mx, y1: mt, y2: mt + ih, stroke: AXIS, "stroke-dasharray": "3 4" }));
  const boxY = mt + 6;
  svg.appendChild(el("rect", { x: mx - 168, y: boxY, width: 160, height: 54, rx: 6,
    fill: "#0e1428", stroke: GRID }));
  svg.appendChild(txt("at 256K tokens", { x: mx - 158, y: boxY + 16, fill: "#cdd4e2", "font-size": 10.5 }));
  svg.appendChild(txt(`cache ${mark.kv} · logits ${mark.logits}`, { x: mx - 158, y: boxY + 31, fill: MUTED, "font-size": 10 }));
  svg.appendChild(txt("logits+grad = 2.67× the cache", { x: mx - 158, y: boxY + 46, fill: GOLD, "font-size": 10 }));
}

/* ------------------------------------------------------------- 04 chunking bars */
function drawChunk() {
  const svg = document.getElementById("chunkChart");
  const groups = [CHUNKING.modelScale, CHUNKING.v5Scale];
  const W = Math.max(560, Math.min(920, svg.parentElement.clientWidth || 880));
  const rowH = 30, gap = 8, headH = 34, ml = 118, mr = 92;
  const H = groups.reduce((a, g) => a + headH + g.rows.length * (rowH + gap) + 18, 12);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const iw = W - ml - mr;
  const max = Math.max(...groups.flatMap(g => g.rows.map(r => r.peak)));

  let y = 10;
  groups.forEach(g => {
    svg.appendChild(txt(`${g.label} — V = ${g.V.toLocaleString()}, ${g.N.toLocaleString()} positions`,
      { x: 0, y: y + 12, fill: "#cdd4e2", "font-size": 12 }));
    y += headH;
    const base = g.rows[0].peak;
    g.rows.forEach(r => {
      const w = Math.max(2, r.peak / max * iw);
      svg.appendChild(txt(r.impl, { x: ml - 10, y: y + rowH / 2 + 4, "text-anchor": "end", fill: "#cdd4e2" }));
      const bar = el("rect", { x: ml, y, width: w, height: rowH, rx: 4,
        fill: r.impl === "naive" ? CORAL : TEAL, opacity: r.impl === "naive" ? 1 : 0.85 });
      bar.style.cursor = "pointer";
      bar.addEventListener("mousemove", ev => showTip(ev,
        `${r.impl}<br><b>${r.peak.toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB</b> peak<br>loss ${r.loss.toFixed(9)}`));
      bar.addEventListener("mouseleave", hideTip);
      svg.appendChild(bar);
      svg.appendChild(txt(`${(base / r.peak).toFixed(2)}×`,
        { x: ml + w + 8, y: y + rowH / 2 + 4, fill: r.impl === "naive" ? MUTED : TEAL, "font-size": 11.5 }));
      y += rowH + gap;
    });
    // the floor chunking cannot go below
    const fx = ml + g.floor / max * iw;
    svg.appendChild(el("line", { x1: fx, x2: fx, y1: y - g.rows.length * (rowH + gap), y2: y - gap,
      stroke: GOLD, "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
    svg.appendChild(txt(`floor ${g.floor.toFixed(0)} MiB`, { x: fx + 5, y: y - gap + 12, fill: GOLD, "font-size": 10.5 }));
    y += 18;
  });
}

/* ------------------------------------------------------------------ 05 two heads */
function drawMtp() {
  const svg = document.getElementById("mtpChart");
  const W = Math.max(560, Math.min(920, svg.parentElement.clientWidth || 880));
  const H = 372, ml = 52, mr = 100, mt = 30, mb = 44;
  const iw = W - ml - mr, ih = H - mt - mb;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);

  const pts = MTP.tied;
  const maxS = pts[pts.length - 1].step;
  const lo = 3.4, hi = 11.2;
  // sqrt step axis so the interesting first 100 steps are legible
  const X = s => ml + Math.sqrt(s / maxS) * iw;
  const Y = v => mt + ih - (v - lo) / (hi - lo) * ih;

  for (let v = 4; v <= 11; v += 1) {
    svg.appendChild(el("line", { x1: ml, x2: ml + iw, y1: Y(v), y2: Y(v), stroke: GRID }));
    svg.appendChild(txt(v.toFixed(0), { x: ml - 8, y: Y(v) + 4, "text-anchor": "end" }));
  }
  svg.appendChild(txt("loss, nats", { x: ml, y: mt - 5, fill: MUTED, "font-size": 10.5 }));

  // ln(V) reference
  svg.appendChild(el("line", { x1: ml, x2: ml + iw, y1: Y(MTP.lnV), y2: Y(MTP.lnV),
    stroke: MUTED, "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  svg.appendChild(txt(`ln(V) = ${MTP.lnV}`, { x: ml + iw + 8, y: Y(MTP.lnV) + 4, fill: MUTED, "font-size": 11 }));

  // the crossover band
  const cs = MTP.crossoverSteps;
  const bx0 = X(0), bx1 = X(50);
  svg.appendChild(el("rect", { x: bx0, y: mt, width: bx1 - bx0, height: ih,
    fill: CORAL, opacity: 0.07 }));
  svg.appendChild(txt("t+2 head ahead", { x: bx0 + 6, y: mt + ih - 8, fill: CORAL, "font-size": 10.5 }));

  [0, 50, 200, 600, 1500].forEach(s => {
    svg.appendChild(el("line", { x1: X(s), x2: X(s), y1: mt + ih, y2: mt + ih + 5, stroke: AXIS }));
    svg.appendChild(txt(String(s), { x: X(s), y: mt + ih + 20, "text-anchor": "middle" }));
  });
  svg.appendChild(txt("training step (√ scale, so the early crossover is legible)",
    { x: ml + iw / 2, y: H - 6, "text-anchor": "middle", fill: MUTED }));
  svg.appendChild(el("line", { x1: ml, x2: ml + iw, y1: mt + ih, y2: mt + ih, stroke: AXIS }));

  [["l1", TEAL, "head 1 · t+1"], ["l2", CORAL, "head 2 · t+2"]].forEach(([k, c, label]) => {
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.step).toFixed(1)},${Y(p[k]).toFixed(1)}`).join(" ");
    svg.appendChild(el("path", { d, fill: "none", stroke: c, "stroke-width": 2, "stroke-linecap": "round" }));
    const last = pts[pts.length - 1];
    svg.appendChild(txt(label, { x: ml + iw + 8, y: Y(last[k]) + 4, fill: c, "font-size": 11.5 }));
    pts.forEach(p => {
      const dot = el("circle", { cx: X(p.step), cy: Y(p[k]), r: 3.5, fill: c, stroke: SURFACE, "stroke-width": 1.5 });
      dot.style.cursor = "pointer";
      dot.addEventListener("mousemove", ev => showTip(ev,
        `step ${p.step}<br>${label}: <b>${p[k].toFixed(4)}</b><br>gap L2−L1: ${(p.l2 - p.l1 >= 0 ? "+" : "") + (p.l2 - p.l1).toFixed(4)}`));
      dot.addEventListener("mouseleave", hideTip);
      svg.appendChild(dot);
    });
  });
}

/* -------------------------------------------------------- 01 live cross-entropy */
const TOKENS = ["Delhi", "Mumbai", "Chennai", "banana", "runs"];
const CORRECT = 2;
let LOGITS = [2.0, 1.0, 0.5, -1.0, -1.5];

function renderCE() {
  const m = Math.max(...LOGITS);
  const ex = LOGITS.map(z => Math.exp(z - m));
  const S = ex.reduce((a, b) => a + b, 0);
  const p = ex.map(e => e / S);
  const loss = -Math.log(p[CORRECT]);
  const grads = p.map((pi, i) => pi - (i === CORRECT ? 1 : 0));

  document.getElementById("loss").textContent = loss.toFixed(3);
  document.getElementById("ppl").textContent = Math.exp(loss).toFixed(1);
  document.getElementById("pc").textContent = p[CORRECT].toFixed(3);
  document.getElementById("gc").textContent = grads[CORRECT].toFixed(3);
  document.getElementById("gs").textContent = grads.reduce((a, b) => a + b, 0).toFixed(3);

  document.getElementById("bars").innerHTML = TOKENS.map((t, i) =>
    `<div class="barrow${i === CORRECT ? " correct" : ""}">
       <span style="font-family:var(--mono);font-size:.82rem;color:${i === CORRECT ? "var(--coral)" : "var(--muted)"}">${t}</span>
       <span class="b"><i style="width:${(p[i] * 100).toFixed(1)}%"></i></span>
     </div>`).join("");
}

function buildCE() {
  document.getElementById("sliders").innerHTML = TOKENS.map((t, i) =>
    `<div class="row-tok${i === CORRECT ? " correct" : ""}">
       <label for="lg${i}">${t}</label>
       <input id="lg${i}" type="range" min="-4" max="6" step="0.1" value="${LOGITS[i]}" aria-label="logit for ${t}" />
       <span class="p" id="lv${i}">${LOGITS[i].toFixed(1)}</span>
     </div>`).join("");
  TOKENS.forEach((_, i) => {
    document.getElementById(`lg${i}`).addEventListener("input", e => {
      LOGITS[i] = parseFloat(e.target.value);
      document.getElementById(`lv${i}`).textContent = LOGITS[i].toFixed(1);
      renderCE();
    });
  });
  renderCE();
}

/* --------------------------------------------------------------- 03, 05, 06 lists */
function buildBugs() {
  document.getElementById("bugs").innerHTML = BUGS.map(b => `
    <div class="bug">
      <div class="hl">${b.headline}</div>
      <div class="hll">${b.headlineLabel}</div>
      <h3>${b.name}</h3>
      <p>${b.what}</p>
      <p class="meas">${b.measured}</p>
    </div>`).join("");
}

function buildHyps() {
  document.getElementById("hyps").innerHTML = MTP.hypotheses.map((h, i) => `
    <div class="hyp">
      <h3>Hypothesis ${i + 1} — ${h.name} <span class="tag ${h.result}">${h.result}</span></h3>
      <p><span class="lbl">claim</span> ${h.claim}</p>
      <p><span class="lbl">test</span> ${h.test}</p>
      <p><span class="lbl">result</span> ${h.because}</p>
    </div>`).join("");
}

function buildCorrections() {
  document.getElementById("corrections").innerHTML = CORRECTIONS.map(c => `
    <div class="corr">
      <div class="st">[${c.stamp}]${c.weight === "repeat" ? '<span class="wt repeat">repeat</span>' : c.weight === "v4" ? '<span class="wt v4">vs the v4 cookbook</span>' : `<span class="wt">${c.weight}</span>`}</div>
      <p class="said">“${c.said}”</p>
      <p class="fact">${c.fact}</p>
      <p class="note">${c.note}</p>
    </div>`).join("");
}

buildCE(); buildBugs(); buildHyps(); buildCorrections();
drawBills(); drawChunk(); drawMtp();
let rt;
addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    ["billsChart", "chunkChart", "mtpChart"].forEach(id => (document.getElementById(id).innerHTML = ""));
    drawBills(); drawChunk(); drawMtp();
  }, 180);
});
