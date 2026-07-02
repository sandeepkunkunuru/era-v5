// A single random seed shared by all four proofs. Changing it regenerates every
// dataset so a fresh "Retrain" converges from different data + different init —
// handy for showing that the result is the phenomenon, not one lucky run.
let seed = 42;
const listeners = [];

export const getSeed = () => seed;

export function setSeed(s) {
  seed = ((s % 100000) + 100000) % 100000; // keep it a tidy non-negative int
  listeners.forEach((f) => f(seed));
}

// deterministic pseudo-random reseed without Math.random tie-in surprises:
// just bump by a large step so consecutive shuffles feel unrelated
export function shuffleSeed() {
  setSeed((seed * 1103515245 + 12345) >>> 8);
}

export const onSeedChange = (f) => listeners.push(f);
