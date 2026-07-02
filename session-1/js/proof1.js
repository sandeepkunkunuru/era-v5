// S1-1 — Activations exist for a reason.
// Same ring data, two models: (a) linear + sigmoid, (b) one ReLU hidden layer.
// Watch the linear boundary stay a straight line (~55%) while ReLU wraps (~99%).
import { makeRings } from './data.js';
import { drawField, drawPoints, clearCanvas } from './viz.js';
import { getSeed, onSeedChange } from './seed.js';

const BOUNDS = { xmin: -3, xmax: 3, ymin: -3, ymax: 3 };

function buildLinear() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.08), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

function buildRelu() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, inputShape: [2], activation: 'relu' }));
  m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.03), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

export function initProof1(root) {
  let X, y;
  const panels = {
    linear: {
      canvas: root.querySelector('#p1-linear-canvas'),
      acc: root.querySelector('#p1-linear-acc'),
      build: buildLinear,
    },
    relu: {
      canvas: root.querySelector('#p1-relu-canvas'),
      acc: root.querySelector('#p1-relu-acc'),
      build: buildRelu,
    },
  };
  const btn = root.querySelector('#p1-train');
  const status = root.querySelector('#p1-status');

  // (re)generate the ring data for the current seed and show the untrained state
  function regen() {
    ({ X, y } = makeRings({ n: 300, seed: getSeed() }));
    for (const key of Object.keys(panels)) {
      clearCanvas(panels[key].canvas);
      drawPoints(panels[key].canvas, X, y, BOUNDS);
      panels[key].acc.textContent = '—';
    }
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
    model.dispose();
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Training…';
    regen(); // fresh sample for the current seed
    const Xt = tf.tensor2d(X);
    const yt = tf.tensor2d(y, [y.length, 1]);
    status.textContent = 'A straight line is trying to separate two rings. It can\'t.';
    await trainOne(panels.linear, 120, Xt, yt);
    status.textContent = 'Now the same data with one ReLU layer — the boundary bends.';
    await trainOne(panels.relu, 120, Xt, yt);
    status.textContent = 'Only the activation changed. Linear stalls; ReLU wraps the ring.';
    Xt.dispose(); yt.dispose();
    btn.disabled = false;
    btn.textContent = 'Retrain';
  });
}
