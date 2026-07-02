import { initProof1 } from './proof1.js';
import { initProof2 } from './proof2.js';
import { initProof3 } from './proof3.js';
import { initProof4 } from './proof4.js';
import { getSeed, setSeed, shuffleSeed, onSeedChange } from './seed.js';

// Wire the shared random-seed control. Changing it reseeds every proof's data.
function wireSeedControl() {
  const input = document.getElementById('seed-input');
  const shuffle = document.getElementById('seed-shuffle');
  if (!input || !shuffle) return;
  input.value = String(getSeed());
  onSeedChange((s) => { input.value = String(s); });
  input.addEventListener('change', () => {
    const v = parseInt(input.value, 10);
    if (Number.isFinite(v)) setSeed(v);
  });
  shuffle.addEventListener('click', () => shuffleSeed());
}

// Give every canvas a crisp, device-pixel-ratio-aware backing store sized to
// however its CSS lays it out.
function sizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
}

async function boot() {
  document.querySelectorAll('canvas').forEach(sizeCanvas);
  wireSeedControl();

  // scroll-reveal for the proof panels (runs regardless of tf availability)
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) if (e.isIntersecting) e.target.classList.add('in');
    },
    { threshold: 0.12 }
  );
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

  if (typeof tf === 'undefined') {
    document.querySelectorAll('.status').forEach((s) => {
      s.textContent = 'TensorFlow.js failed to load — check your connection and refresh.';
    });
    return;
  }
  // tiny nets: cpu is deterministic and avoids webgl warmup jank. Await so no
  // training call can race ahead of the backend being ready.
  try { await tf.setBackend('cpu'); } catch (e) { /* fall back to default backend */ }
  await tf.ready();

  initProof1(document.getElementById('proof-1'));
  initProof2(document.getElementById('proof-2'));
  initProof3(document.getElementById('proof-3'));
  initProof4(document.getElementById('proof-4'));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
