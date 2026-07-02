// S1-2 — Depth without nonlinearity is a lie.
// 1 linear layer vs 5 stacked linear layers vs 5 layers + ReLU, on the ring data.
// The 1-layer and 5-linear boundaries are the SAME line. Bonus: multiply the five
// weight matrices and show the product is one 2x1 map.
import { makeRings } from './data.js';
import { drawField, drawPoints, clearCanvas } from './viz.js';
import { getSeed, onSeedChange } from './seed.js';

const BOUNDS = { xmin: -3, xmax: 3, ymin: -3, ymax: 3 };

function buildOneLinear() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.08), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

// Five dense layers, NO activation between them; only the final sigmoid squashes.
function buildFiveLinear() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 8, inputShape: [2], activation: 'linear' }));
  m.add(tf.layers.dense({ units: 8, activation: 'linear' }));
  m.add(tf.layers.dense({ units: 8, activation: 'linear' }));
  m.add(tf.layers.dense({ units: 8, activation: 'linear' }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.05), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

function buildFiveRelu() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 8, inputShape: [2], activation: 'relu' }));
  m.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.02), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

// Collapse the 5 linear layers into one effective 2x1 map: W = K1·K2·K3·K4·K5.
function collapse(model) {
  return tf.tidy(() => {
    const kernels = [];
    for (const layer of model.layers) kernels.push(layer.getWeights()[0]);
    let W = kernels[0];
    for (let i = 1; i < kernels.length; i++) W = tf.matMul(W, kernels[i]);
    return W.dataSync(); // shape [2,1] -> 2 numbers
  });
}

export function initProof2(root) {
  let X, y;
  const panels = {
    one: { canvas: root.querySelector('#p2-one-canvas'), acc: root.querySelector('#p2-one-acc'), build: buildOneLinear },
    five: { canvas: root.querySelector('#p2-five-canvas'), acc: root.querySelector('#p2-five-acc'), build: buildFiveLinear },
    relu: { canvas: root.querySelector('#p2-relu-canvas'), acc: root.querySelector('#p2-relu-acc'), build: buildFiveRelu },
  };
  const btn = root.querySelector('#p2-train');
  const status = root.querySelector('#p2-status');
  const matrixOut = root.querySelector('#p2-matrix');

  function regen() {
    ({ X, y } = makeRings({ n: 300, seed: getSeed() }));
    for (const key of Object.keys(panels)) {
      clearCanvas(panels[key].canvas);
      drawPoints(panels[key].canvas, X, y, BOUNDS);
      panels[key].acc.textContent = '—';
    }
    matrixOut.textContent = '';
  }
  regen();
  onSeedChange(() => { if (!btn.disabled) regen(); });

  async function trainOne(panel, epochs, Xt, yt) {
    const model = panel.build();
    const predictFn = (t) => model.predict(t).reshape([-1]);
    await model.fit(Xt, yt, {
      epochs,
      batchSize: 32,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (epoch % 3 === 0 || epoch === epochs - 1) {
            drawField(panel.canvas, predictFn, BOUNDS);
            drawPoints(panel.canvas, X, y, BOUNDS);
            panel.acc.textContent = (logs.acc * 100).toFixed(1) + '%';
            await tf.nextFrame();
          }
        },
      },
    });
    return model;
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Training…';
    regen(); // fresh sample for the current seed
    const Xt = tf.tensor2d(X);
    const yt = tf.tensor2d(y, [y.length, 1]);
    status.textContent = 'One linear layer: a line.';
    const m1 = await trainOne(panels.one, 120, Xt, yt);
    m1.dispose();
    status.textContent = 'Five linear layers, no activations: still the exact same line.';
    const m5 = await trainOne(panels.five, 150, Xt, yt);
    const W = collapse(m5);
    m5.dispose();
    status.textContent = 'Add ReLU between the same five layers — the tie breaks.';
    const mr = await trainOne(panels.relu, 160, Xt, yt);
    mr.dispose();
    Xt.dispose(); yt.dispose();

    matrixOut.textContent =
      'K1·K2·K3·K4·K5  =  [ ' + W[0].toFixed(4) + ' , ' + W[1].toFixed(4) + ' ]ᵀ\n' +
      'Five weight matrices (2×8·8×8·8×8·8×8·8×1) multiply down to ONE 2×1 map.\n' +
      'That is why 1 linear layer and 5 linear layers can only ever draw a straight line.';
    status.textContent = 'Depth without nonlinearity added nothing. ReLU is what buys capacity.';
    btn.disabled = false;
    btn.textContent = 'Retrain';
  });
}
