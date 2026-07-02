// S1-3 — Embeddings learn similarity from nothing but next-token.
// Train a tiny embedding -> softmax next-token model on a toy grammar where
// same-category tokens share next-token distributions. Project embeddings to 2D
// (PCA) and watch the categories cluster — similarity was never supplied.
import { makeGrammarPairs, buildVocab } from './data.js';
import { drawEmbedding } from './viz.js';
import { getSeed, onSeedChange } from './seed.js';

const CAT_COLORS = {
  animal: '#2DD4BF',
  fruit: '#FB7185',
  verb: '#FBBF24',
  structural: '#8A93A8',
};

// PCA to 2D via power iteration on the covariance (embedding dim is tiny).
function pca2(rows) {
  const V = rows.length, D = rows[0].length;
  const mean = new Array(D).fill(0);
  for (const r of rows) for (let d = 0; d < D; d++) mean[d] += r[d] / V;
  const Xc = rows.map((r) => r.map((v, d) => v - mean[d]));
  // covariance D x D
  const C = Array.from({ length: D }, () => new Array(D).fill(0));
  for (const r of Xc) for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) C[i][j] += (r[i] * r[j]) / V;

  const matVec = (M, v) => M.map((row) => row.reduce((s, m, k) => s + m * v[k], 0));
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  const powerIter = (M) => {
    let v = new Array(D).fill(0).map((_, i) => Math.sin(i + 1)); // deterministic start
    for (let it = 0; it < 100; it++) {
      let w = matVec(M, v);
      const n = norm(w);
      v = w.map((x) => x / n);
    }
    return v;
  };
  const v1 = powerIter(C);
  // deflate: C' = C - λ1 v1 v1ᵀ
  const Cv1 = matVec(C, v1);
  const lam1 = v1.reduce((s, x, i) => s + x * Cv1[i], 0);
  const C2 = C.map((row, i) => row.map((val, j) => val - lam1 * v1[i] * v1[j]));
  const v2 = powerIter(C2);

  return Xc.map((r) => ({
    x: r.reduce((s, x, i) => s + x * v1[i], 0),
    y: r.reduce((s, x, i) => s + x * v2[i], 0),
  }));
}

export function initProof3(root) {
  const { words, catOf } = buildVocab();
  const V = words.length;
  const EMB = 8;

  const canvas = root.querySelector('#p3-canvas');
  const btn = root.querySelector('#p3-train');
  const status = root.querySelector('#p3-status');
  const nn = root.querySelector('#p3-nn');

  let model, inputs, targets;

  // fresh grammar sample + fresh (random) model for the current seed
  function rebuild() {
    if (model) model.dispose();
    ({ inputs, targets } = makeGrammarPairs({ nSentences: 900, seed: getSeed() }));
    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: V, outputDim: EMB, inputLength: 1 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: V, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.05), loss: 'sparseCategoricalCrossentropy' });
  }

  function currentEmbeddings() {
    const w = model.layers[0].getWeights()[0];
    const arr = w.arraySync(); // [V, EMB]
    return arr;
  }

  function render() {
    const emb = currentEmbeddings();
    const proj = pca2(emb);
    const points = words.map((w, i) => ({
      x: proj[i].x,
      y: proj[i].y,
      label: w,
      cat: catOf[w],
    }));
    drawEmbedding(canvas, points, { catColors: CAT_COLORS });
    // nearest-neighbour same-category score (exclude structural tokens)
    let same = 0, total = 0;
    for (let i = 0; i < V; i++) {
      if (catOf[words[i]] === 'structural') continue;
      total++;
      let best = -1, bestD = Infinity;
      for (let j = 0; j < V; j++) {
        if (i === j) continue;
        let d = 0;
        for (let k = 0; k < EMB; k++) { const df = emb[i][k] - emb[j][k]; d += df * df; }
        if (d < bestD) { bestD = d; best = j; }
      }
      if (catOf[words[best]] === catOf[words[i]]) same++;
    }
    nn.textContent = `same-category nearest neighbour: ${same}/${total} tokens`;
    return same / total;
  }

  // show the untrained (random) embedding so the "before" is visible
  function showUntrained() {
    rebuild();
    render();
    nn.textContent = 'untrained: embeddings are random — no structure yet';
  }
  showUntrained();
  onSeedChange(() => { if (!btn.disabled) showUntrained(); });

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Training…';
    rebuild(); // fresh grammar + fresh random init for the current seed
    render();
    status.textContent = 'Training only to predict the next token. Nothing tells it what is similar.';
    const Xt = tf.tensor2d(inputs, [inputs.length, 1], 'int32');
    // sparseCategoricalCrossentropy in tfjs-layers requires float32 labels
    const yt = tf.tensor1d(targets, 'float32');
    await model.fit(Xt, yt, {
      epochs: 120,
      batchSize: 64,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch) => {
          if (epoch % 4 === 0 || epoch === 119) {
            render();
            await tf.nextFrame();
          }
        },
      },
    });
    Xt.dispose(); yt.dispose();
    const score = render();
    status.textContent =
      score > 0.9
        ? 'Emergent clustering: same-category tokens landed together — learned purely from next-token.'
        : 'Categories separate as training proceeds — learned purely from next-token.';
    btn.disabled = false;
    btn.textContent = 'Retrain';
  });
}
