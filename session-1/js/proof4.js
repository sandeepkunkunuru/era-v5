// S1-4 — Memorization vs generalization, and data closes the gap.
// A high-capacity net trained on 20 / 200 / 2000 points from the same noisy
// distribution, all evaluated on one big held-out set. Tiny data -> train≈100%,
// test poor (memorized noise). More data -> the gap shrinks.
import { makeMoons } from './data.js';
import { drawBars, COLORS } from './viz.js';
import { getSeed, onSeedChange } from './seed.js';

const SIZES = [20, 200, 2000];

function buildBig() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 64, inputShape: [2], activation: 'relu' }));
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

function accuracy(model, Xt, yArr) {
  return tf.tidy(() => {
    const p = model.predict(Xt).reshape([-1]).dataSync();
    let correct = 0;
    for (let i = 0; i < yArr.length; i++) if ((p[i] > 0.5 ? 1 : 0) === yArr[i]) correct++;
    return correct / yArr.length;
  });
}

export function initProof4(root) {
  const canvas = root.querySelector('#p4-canvas');
  const btn = root.querySelector('#p4-train');
  const status = root.querySelector('#p4-status');
  const gapOut = root.querySelector('#p4-gap');

  // one large held-out test set, shared across all three runs (seed-derived)
  let test, Xtest;
  function regenTest() {
    if (Xtest) Xtest.dispose();
    test = makeMoons({ n: 2000, seed: getSeed() + 999, noise: 0.18, flip: 0.08 });
    Xtest = tf.tensor2d(test.X);
  }
  regenTest();

  const trainAccs = SIZES.map(() => 0);
  const testAccs = SIZES.map(() => 0);

  function renderBars() {
    drawBars(
      canvas,
      SIZES.map((s) => String(s)),
      [
        { label: 'train', values: trainAccs, color: `rgb(${COLORS.class0.join(',')})` },
        { label: 'test', values: testAccs, color: `rgb(${COLORS.class1.join(',')})` },
      ],
      { max: 1 }
    );
  }
  renderBars();

  // reset the chart when the seed changes (unless a run is in progress)
  onSeedChange(() => {
    if (btn.disabled) return;
    regenTest();
    for (let i = 0; i < SIZES.length; i++) { trainAccs[i] = 0; testAccs[i] = 0; }
    gapOut.textContent = '';
    renderBars();
  });

  async function runSize(size, idx) {
    const model = buildBig();
    const train = makeMoons({ n: size, seed: getSeed() * 10 + 100 + idx, noise: 0.18, flip: 0.08 });
    const Xt = tf.tensor2d(train.X);
    const yt = tf.tensor2d(train.y, [train.y.length, 1]);
    const epochs = size <= 20 ? 300 : size <= 200 ? 200 : 80;
    await model.fit(Xt, yt, {
      epochs,
      batchSize: Math.min(32, size),
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (epoch % 20 === 0 || epoch === epochs - 1) {
            trainAccs[idx] = logs.acc;
            testAccs[idx] = accuracy(model, Xtest, test.y);
            renderBars();
            await tf.nextFrame();
          }
        },
      },
    });
    trainAccs[idx] = accuracy(model, Xt, train.y);
    testAccs[idx] = accuracy(model, Xtest, test.y);
    renderBars();
    Xt.dispose(); yt.dispose(); model.dispose();
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Training…';
    regenTest(); // fresh held-out set for the current seed
    gapOut.textContent = '';
    for (let i = 0; i < SIZES.length; i++) {
      status.textContent = `Training the same 64×64 net on ${SIZES[i]} points…`;
      trainAccs[i] = 0; testAccs[i] = 0;
      await runSize(SIZES[i], i);
      const gap = (trainAccs[i] - testAccs[i]) * 100;
      status.textContent = `${SIZES[i]} points → train ${(trainAccs[i] * 100).toFixed(0)}%, test ${(testAccs[i] * 100).toFixed(0)}% (gap ${gap.toFixed(0)} pts)`;
    }
    const gaps = SIZES.map((_, i) => ((trainAccs[i] - testAccs[i]) * 100));
    gapOut.textContent =
      `generalization gap:  20 → ${gaps[0].toFixed(0)} pts   ·   200 → ${gaps[1].toFixed(0)} pts   ·   2000 → ${gaps[2].toFixed(0)} pts`;
    status.textContent = 'Same model, same noise. The only thing that closed the gap was more data.';
    btn.disabled = false;
    btn.textContent = 'Retrain';
  });
}
