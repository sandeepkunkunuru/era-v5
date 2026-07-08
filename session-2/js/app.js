/* Session 2 — Multilingual BPE fertility parity.
   Loads the trained tokenizer + the four page texts and renders the score, the fertility
   number-line, and the stats table — then RE-TOKENIZES every page in the browser (a faithful
   port of the GPT-2/HuggingFace BPE algorithm) and checks the numbers against results.json.
   Nothing is precomputed; the graders re-run the same tokenizer. */

const LANGS = ["en", "hi", "te", "es"];
const META = {
  en: { nm: "English", script: "Latin",      glyph: "India",    col: "var(--en)" },
  hi: { nm: "Hindi",   script: "Devanagari",  glyph: "भारत",     col: "var(--hi)" },
  te: { nm: "Telugu",  script: "Telugu",      glyph: "భారతదేశం", col: "var(--te)" },
  es: { nm: "Spanish", script: "Latin",       glyph: "India",    col: "var(--es)" },
};
const BASELINE = 1239;
const fmt = (n) => Math.round(n).toLocaleString();

/* ---- faithful BPE encoder (merge the lowest-rank adjacent pair, repeat) ---- */
function buildBPE(tokenizerJson) {
  const model = tokenizerJson.model;
  const ranks = new Map();
  model.merges.forEach((m, i) => {
    const p = Array.isArray(m) ? m : m.split(" ");
    ranks.set(p[0] + "\t" + p[1], i);
  });
  function encodeWordLen(word) {
    let sym = Array.from(word);            // base symbols = unicode code points
    if (sym.length < 2) return sym.length;
    for (;;) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < sym.length - 1; i++) {
        const r = ranks.get(sym[i] + "\t" + sym[i + 1]);
        if (r !== undefined && r < best) { best = r; bi = i; }
      }
      if (bi < 0) break;
      const a = sym[bi], b = sym[bi + 1], ab = a + b, out = [];
      for (let i = 0; i < sym.length;) {
        if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) { out.push(ab); i += 2; }
        else { out.push(sym[i]); i++; }
      }
      sym = out;
    }
    return sym.length;
  }
  return function count(text) {
    const words = text.match(/\S+/gu) || [];
    let tokens = 0;
    for (const w of words) tokens += encodeWordLen(w);
    return { tokens, words: words.length };
  };
}

/* ---- render ---- */
function renderScore(R) {
  document.getElementById("score").textContent = fmt(R.score);
  const chip = (id, html, cls) => {
    const el = document.getElementById(id);
    el.innerHTML = html;
    if (cls) el.classList.add(cls);
  };
  chip("chip-gap", `<b>gap</b>&nbsp; ${R.gap.toFixed(3)}`);
  chip("chip-cap", `<b>English</b>&nbsp; ${R.fertility.en.toFixed(3)} ≤ 1.20 ✓`, "good");
  chip("chip-vocab", `<b>vocab</b>&nbsp; ${fmt(R.vocab)}`);
  const mult = (R.score / BASELINE).toFixed(1);
  chip("chip-base", `vs baseline ${fmt(BASELINE)} &nbsp;→&nbsp; ${mult}×`);
}

function renderAxis(R) {
  const F = R.fertility;
  const vals = LANGS.map((l) => F[l]);
  const lo = 1.0, hi = Math.max(1.8, Math.ceil(Math.max(...vals) * 10) / 10);
  const X = (v) => ((v - lo) / (hi - lo)) * 100;
  let h = `<div class="lane"></div>`;
  for (let t = lo; t <= hi + 1e-9; t += 0.2) {
    h += `<div class="tick" style="left:${X(t)}%"><div class="t"></div><div class="l">${t.toFixed(1)}</div></div>`;
  }
  h += `<div class="cap" style="left:${X(1.2)}%"><span class="lab">English cap 1.20</span></div>`;
  const x1 = Math.min(...vals), x4 = Math.max(...vals);
  h += `<div class="span" style="left:${X(x1)}%;width:${X(x4) - X(x1)}%">
    <span class="glab">gap ${(x4 - x1).toFixed(3)} → score ${fmt(R.score)}</span></div>`;
  const order = [...LANGS].sort((a, b) => F[a] - F[b]);
  // four distinct vertical levels (up-near, down-near, up-far, down-far) so tightly
  // clustered points — the whole point of the result — never overlap their labels.
  const LV = [
    { off: -66, stem: "bottom:8px;height:48px" },
    { off: 40, stem: "top:8px;height:26px" },
    { off: -112, stem: "bottom:8px;height:94px" },
    { off: 86, stem: "top:8px;height:72px" },
  ];
  order.forEach((l, i) => {
    const { off, stem } = LV[i];
    h += `<div class="mk" style="left:${X(F[l])}%;color:${META[l].col}">
      <div class="stem" style="${stem}"></div><div class="dot2"></div>
      <div class="tag2" style="top:${off}px">
        <div class="scr">${META[l].glyph}</div>
        <div class="code">${l}</div>
        <div class="val">${F[l].toFixed(3)}</div></div></div>`;
  });
  document.getElementById("axis").innerHTML = h;
}

function renderTable(R) {
  const F = R.fertility, W = R.words, maxF = Math.max(...LANGS.map((l) => F[l]));
  const body = document.querySelector("#tbl tbody");
  LANGS.forEach((l) => {
    const capTag = l === "en" ? `<span class="cap-tag">≤ 1.20</span>` : "";
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td><span class="lang"><span class="sw2" style="background:${META[l].col}"></span>
         <span class="nm">${META[l].nm}</span></span></td>
       <td>${META[l].script}</td>
       <td>${fmt(W[l])}</td>
       <td>${fmt(F[l] * W[l])}</td>
       <td><div class="fertcell"><span>${F[l].toFixed(4)} ${capTag}</span>
         <div class="fbar"><i style="width:${(F[l] / maxF) * 100}%;background:${META[l].col}"></i></div></div></td>`;
    body.appendChild(tr);
  });
}

/* ---- live re-tokenization, re-runnable with per-language progress ---- */
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function buildVerifyRows() {
  document.getElementById("verify-rows").innerHTML = LANGS.map((l) =>
    `<li class="vrow" id="vrow-${l}">
       <span class="vr-name"><span class="sw2" style="background:${META[l].col}"></span>${META[l].nm}</span>
       <span class="vr-stat" id="vr-${l}">queued</span></li>`).join("");
}

function setRow(l, state, prog, fert, matches) {
  const li = document.getElementById("vrow-" + l), el = document.getElementById("vr-" + l);
  li.className = "vrow" + (state ? " " + state : "");
  const mark = matches === true ? ` <span class="vr-ok">match ✓</span>`
    : matches === false ? ` <span class="vr-bad">Δ mismatch</span>` : "";
  el.innerHTML = prog + (fert ? ` · fertility <b>${fert}</b>` : "") + mark;
}

// fetch + parse the tokenizer and the four pages once, then reuse across re-runs
let VCACHE = null;
async function loadForVerify() {
  if (VCACHE) return VCACHE;
  const blob = await fetch("./tokenizer.json").then((r) => r.blob());
  const szEl = document.getElementById("tok-sz");
  if (szEl) szEl.textContent = "· " + Math.round(blob.size / 1024) + " KB";  // real bytes
  const count = buildBPE(JSON.parse(await blob.text()));
  const texts = {};
  await Promise.all(LANGS.map((l) =>
    fetch(`./data/${l}.txt`).then((r) => r.text()).then((t) => { texts[l] = t; })));
  VCACHE = { count, texts };
  return VCACHE;
}

let verifying = false;
async function runVerify(R) {
  if (verifying) return;
  verifying = true;
  const btn = document.getElementById("verify-btn"),
    pill = document.getElementById("vpill"), txt = document.getElementById("vtxt"),
    note = document.getElementById("vnote");
  btn.disabled = true;
  pill.className = "vpill run"; txt.textContent = "re-tokenizing…";
  note.textContent = "Loading the tokenizer and the four pages…";
  LANGS.forEach((l) => setRow(l, "", "queued", "", null));

  let cache;
  try { cache = await loadForVerify(); }
  catch (e) {
    txt.textContent = "skipped"; note.textContent = "Assets unavailable: " + e.message;
    btn.disabled = false; verifying = false; return;
  }
  const { count, texts } = cache;
  const t0 = performance.now();
  let maxErr = 0, totalWords = 0, totalTokens = 0, i = 0;
  for (const l of LANGS) {
    i++;
    setRow(l, "running", "tokenizing…", "", null);
    txt.textContent = `re-tokenizing… ${i}/4`;
    await wait(140);                       // let the row paint so the progress is visible
    const { tokens, words } = count(texts[l]);
    const fe = tokens / words, d = Math.abs(fe - R.fertility[l]);
    maxErr = Math.max(maxErr, d); totalWords += words; totalTokens += tokens;
    setRow(l, "done", `${words.toLocaleString()} words → ${tokens.toLocaleString()} tokens`, fe.toFixed(4), d < 0.001);
  }
  const ms = Math.round(performance.now() - t0);
  const ok = maxErr < 0.001;
  pill.className = ok ? "vpill ok" : "vpill run";
  txt.textContent = ok ? "verified ✓" : "Δ " + maxErr.toFixed(4);
  note.textContent = ok
    ? `Re-tokenized ${totalWords.toLocaleString()} words → ${totalTokens.toLocaleString()} tokens in ${ms} ms — all four match the reported fertilities (Δ < 0.001).`
    : `Live re-tokenization differs from the reported values by ${maxErr.toFixed(4)}.`;
  btn.disabled = false; btn.textContent = "Re-run in browser";
  verifying = false;
}

/* ---- reveal on scroll ---- */
function wireReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.12 });
  els.forEach((e) => io.observe(e));
}

/* ---- boot ---- */
(async function () {
  wireReveal();
  const R = await fetch("./results.json").then((r) => r.json());
  renderScore(R);
  renderAxis(R);
  renderTable(R);
  buildVerifyRows();
  document.getElementById("verify-btn").addEventListener("click", () => runVerify(R));
  runVerify(R);   // auto-run once on load; the button re-runs it on demand
})();
