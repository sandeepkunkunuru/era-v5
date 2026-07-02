// Headless validation of the four proofs. Mirrors the exact model architectures
// used in the browser (assets/js/proof*.js) and prints the numbers that the page
// is claiming. Run: node verify/run.mjs
import * as tf from '@tensorflow/tfjs';
import { makeRings, makeMoons, makeGrammarPairs, buildVocab } from '../session-1/js/data.js';

await tf.setBackend('cpu');
const acc = (a) => (a * 100).toFixed(1) + '%';

function evalAcc(model, X, y) {
  return tf.tidy(() => {
    const p = model.predict(tf.tensor2d(X)).reshape([-1]).dataSync();
    let c = 0;
    for (let i = 0; i < y.length; i++) if ((p[i] > 0.5 ? 1 : 0) === y[i]) c++;
    return c / y.length;
  });
}

// ---------- S1-1 ----------
async function proof1() {
  const { X, y } = makeRings({ n: 300, seed: 7 });
  const Xt = tf.tensor2d(X), yt = tf.tensor2d(y, [y.length, 1]);

  const lin = tf.sequential();
  lin.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
  lin.compile({ optimizer: tf.train.adam(0.08), loss: 'binaryCrossentropy' });
  await lin.fit(Xt, yt, { epochs: 120, batchSize: 32, shuffle: true, verbose: 0 });

  const relu = tf.sequential();
  relu.add(tf.layers.dense({ units: 16, inputShape: [2], activation: 'relu' }));
  relu.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  relu.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  relu.compile({ optimizer: tf.train.adam(0.03), loss: 'binaryCrossentropy' });
  await relu.fit(Xt, yt, { epochs: 120, batchSize: 32, shuffle: true, verbose: 0 });

  console.log(`S1-1  linear=${acc(evalAcc(lin, X, y))}   relu=${acc(evalAcc(relu, X, y))}   (expect linear~50-60%, relu~95-100%)`);
}

// ---------- S1-2 ----------
async function proof2() {
  const { X, y } = makeRings({ n: 300, seed: 7 });
  const Xt = tf.tensor2d(X), yt = tf.tensor2d(y, [y.length, 1]);

  const one = tf.sequential();
  one.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
  one.compile({ optimizer: tf.train.adam(0.08), loss: 'binaryCrossentropy' });
  await one.fit(Xt, yt, { epochs: 120, batchSize: 32, shuffle: true, verbose: 0 });

  const five = tf.sequential();
  for (let i = 0; i < 4; i++) five.add(tf.layers.dense({ units: 8, inputShape: i === 0 ? [2] : undefined, activation: 'linear' }));
  five.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  five.compile({ optimizer: tf.train.adam(0.05), loss: 'binaryCrossentropy' });
  await five.fit(Xt, yt, { epochs: 150, batchSize: 32, shuffle: true, verbose: 0 });

  const relu = tf.sequential();
  for (let i = 0; i < 4; i++) relu.add(tf.layers.dense({ units: 8, inputShape: i === 0 ? [2] : undefined, activation: 'relu' }));
  relu.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  relu.compile({ optimizer: tf.train.adam(0.02), loss: 'binaryCrossentropy' });
  await relu.fit(Xt, yt, { epochs: 160, batchSize: 32, shuffle: true, verbose: 0 });

  // collapse the 5 linear kernels
  const W = tf.tidy(() => {
    let m = five.layers[0].getWeights()[0];
    for (let i = 1; i < five.layers.length; i++) m = tf.matMul(m, five.layers[i].getWeights()[0]);
    return m.dataSync();
  });
  console.log(`S1-2  1-layer=${acc(evalAcc(one, X, y))}   5-linear=${acc(evalAcc(five, X, y))}   5-relu=${acc(evalAcc(relu, X, y))}`);
  console.log(`      collapsed K1..K5 = [${W[0].toFixed(4)}, ${W[1].toFixed(4)}]  (5 matrices -> one 2x1 map; expect 1-layer==5-linear both low, relu high)`);
}

// ---------- S1-3 ----------
async function proof3() {
  const { words, catOf } = buildVocab();
  const { inputs, targets } = makeGrammarPairs({ nSentences: 900, seed: 3 });
  const V = words.length, EMB = 8;
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: V, outputDim: EMB, inputLength: 1 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: V, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(0.05), loss: 'sparseCategoricalCrossentropy' });
  await model.fit(tf.tensor2d(inputs, [inputs.length, 1], 'int32'), tf.tensor1d(targets, 'float32'),
    { epochs: 120, batchSize: 64, shuffle: true, verbose: 0 });

  const emb = model.layers[0].getWeights()[0].arraySync();
  let same = 0, total = 0;
  for (let i = 0; i < V; i++) {
    if (catOf[words[i]] === 'structural') continue;
    total++;
    let best = -1, bd = Infinity;
    for (let j = 0; j < V; j++) {
      if (i === j) continue;
      let d = 0; for (let k = 0; k < EMB; k++) { const df = emb[i][k] - emb[j][k]; d += df * df; }
      if (d < bd) { bd = d; best = j; }
    }
    if (catOf[words[best]] === catOf[words[i]]) same++;
  }
  console.log(`S1-3  same-category nearest-neighbour = ${same}/${total}  (expect all/most; emergent clustering)`);
}

// ---------- S1-4 ----------
async function proof4() {
  const test = makeMoons({ n: 2000, seed: 999, noise: 0.18, flip: 0.08 });
  const sizes = [20, 200, 2000];
  const out = [];
  for (let idx = 0; idx < sizes.length; idx++) {
    const size = sizes[idx];
    const train = makeMoons({ n: size, seed: 100 + idx, noise: 0.18, flip: 0.08 });
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 64, inputShape: [2], activation: 'relu' }));
    m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy' });
    const epochs = size <= 20 ? 300 : size <= 200 ? 200 : 80;
    await m.fit(tf.tensor2d(train.X), tf.tensor2d(train.y, [train.y.length, 1]), { epochs, batchSize: Math.min(32, size), shuffle: true, verbose: 0 });
    const tr = evalAcc(m, train.X, train.y), te = evalAcc(m, test.X, test.y);
    out.push(`${size}: train=${acc(tr)} test=${acc(te)} gap=${((tr - te) * 100).toFixed(0)}pts`);
  }
  console.log('S1-4  ' + out.join('   |   ') + '   (expect gap large@20 -> small@2000)');
}

await proof1();
await proof2();
await proof3();
await proof4();
console.log('\nDone. Backend:', tf.getBackend());
