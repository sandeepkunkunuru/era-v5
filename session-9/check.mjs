/* Assert the page's numbers still match the harness run. Run after any re-run:
     node check.mjs                                                            */
import { readFileSync } from "node:fs";
import { CHUNKING, MTP } from "./data.js";

const R = JSON.parse(readFileSync("../session-09-loss-harness/results.json", "utf8"));
const fails = [];
const eq = (a, b, tol, what) => { if (Math.abs(a - b) > tol) fails.push(`${what}: page ${a} vs run ${b}`); };

[["modelScale", "memory_model_scale"], ["v5Scale", "memory_v5_vocab"]].forEach(([pk, rk]) => {
  const p = CHUNKING[pk], r = R[rk];
  eq(p.N, r.N, 0, `${pk}.N`); eq(p.V, r.V, 0, `${pk}.V`);
  eq(p.floor, r.floor_mib, 0.05, `${pk}.floor`);
  eq(p.ratio, r.ratio, 0.005, `${pk}.ratio`);
  if (p.rows.length !== r.rows.length) fails.push(`${pk}: row count`);
  p.rows.forEach((row, i) => {
    eq(row.peak, r.rows[i].peak_mib, 0.05, `${pk}.rows[${i}].peak`);
    eq(row.loss, r.rows[i].loss, 1e-8, `${pk}.rows[${i}].loss`);
  });
});

const h = R.part2.history;
if (MTP.tied.length !== h.length) fails.push("MTP: point count");
MTP.tied.forEach((p, i) => {
  eq(p.step, h[i].step, 0, `MTP[${i}].step`);
  eq(p.l1, h[i].l1, 5e-4, `MTP[${i}].l1`);
  eq(p.l2, h[i].l2, 5e-4, `MTP[${i}].l2`);
});
eq(MTP.lnV, R.part2.ln_V, 5e-4, "MTP.lnV");
eq(MTP.unigramEntropy, R.part2.unigram_entropy, 5e-4, "MTP.unigramEntropy");
if (JSON.stringify(MTP.crossoverSteps) !== JSON.stringify(R.part2.negative_gap_steps))
  fails.push(`crossoverSteps: ${MTP.crossoverSteps} vs ${R.part2.negative_gap_steps}`);

if (fails.length) { console.error("MISMATCH:\n  " + fails.join("\n  ")); process.exit(1); }
console.log(`data.js agrees with results.json — ${CHUNKING.modelScale.rows.length + CHUNKING.v5Scale.rows.length} memory rows, ${MTP.tied.length} training points`);
