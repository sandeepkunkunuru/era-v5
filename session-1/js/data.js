// Synthetic datasets used across the four proofs.
// All generators are deterministic given a seed so the "money shots" reproduce.

// --- tiny seedable RNG (mulberry32) ---
export function rng(seed = 42) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// standard normal via Box-Muller, driven by a uniform rng
function gauss(r) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Two concentric noisy rings. Inner ring = class 0, outer ring = class 1.
// Not linearly separable — no straight line can split an inner disc-ring from
// an outer ring around it.
export function makeRings({ n = 300, seed = 7, innerR = 1.0, outerR = 2.2, noise = 0.16 } = {}) {
  const r = rng(seed);
  const X = [];
  const y = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const cls = i < half ? 0 : 1;
    const base = cls === 0 ? innerR : outerR;
    const theta = r() * 2 * Math.PI;
    const rad = base + gauss(r) * noise;
    X.push([rad * Math.cos(theta), rad * Math.sin(theta)]);
    y.push(cls);
  }
  return { X, y };
}

// A "learnable" boundary (two moons) with controllable label noise, plus a
// held-out set drawn from the same distribution. Used for the data-scaling proof.
export function makeMoons({ n = 200, seed = 1, noise = 0.18, flip = 0.08 } = {}) {
  const r = rng(seed);
  const X = [];
  const y = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const cls = i < half ? 0 : 1;
    const t = r() * Math.PI;
    let px, py;
    if (cls === 0) {
      px = Math.cos(t);
      py = Math.sin(t);
    } else {
      px = 1 - Math.cos(t);
      py = 0.5 - Math.sin(t);
    }
    px += gauss(r) * noise;
    py += gauss(r) * noise;
    // label noise: flip a fraction of labels so tiny datasets can "memorize" it
    const label = r() < flip ? 1 - cls : cls;
    X.push([px, py]);
    y.push(label);
  }
  return { X, y };
}

// ---- Toy grammar for the embedding proof ----
// Categories share next-token distributions by construction, so a next-token
// model must give same-category tokens similar embeddings.
export const GRAMMAR = {
  categories: {
    animal: ['cat', 'dog', 'cow', 'fox'],
    fruit: ['apple', 'mango', 'pear', 'plum'],
    verb: ['eat', 'chase', 'see', 'like'],
  },
  structural: ['the'],
  // templates written as category slots; same-category tokens are interchangeable
  templates: [
    ['the', 'animal', 'verb', 'the', 'fruit'],
    ['the', 'animal', 'verb', 'the', 'animal'],
  ],
};

export function buildVocab() {
  const words = [];
  for (const c of Object.keys(GRAMMAR.categories)) words.push(...GRAMMAR.categories[c]);
  words.push(...GRAMMAR.structural);
  const stoi = {};
  words.forEach((w, i) => (stoi[w] = i));
  const catOf = {};
  for (const c of Object.keys(GRAMMAR.categories)) {
    for (const w of GRAMMAR.categories[c]) catOf[w] = c;
  }
  for (const w of GRAMMAR.structural) catOf[w] = 'structural';
  return { words, stoi, catOf };
}

// Generate (currentToken -> nextToken) training pairs by sampling templates.
export function makeGrammarPairs({ nSentences = 800, seed = 3 } = {}) {
  const r = rng(seed);
  const { stoi, catOf } = buildVocab();
  const pick = (cat) => {
    const arr = GRAMMAR.categories[cat];
    return arr[Math.floor(r() * arr.length)];
  };
  const inputs = [];
  const targets = [];
  for (let s = 0; s < nSentences; s++) {
    const tpl = GRAMMAR.templates[Math.floor(r() * GRAMMAR.templates.length)];
    const sentence = tpl.map((slot) =>
      GRAMMAR.categories[slot] ? pick(slot) : slot
    );
    for (let i = 0; i < sentence.length - 1; i++) {
      inputs.push(stoi[sentence[i]]);
      targets.push(stoi[sentence[i + 1]]);
    }
  }
  return { inputs, targets, stoi, catOf };
}
