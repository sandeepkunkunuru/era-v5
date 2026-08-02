# TDES — Training Data Execution System (ERA V5 · Session 6)

A small but complete, **fully deterministic** data-execution system for LLM training.
It implements the whole path Session 6 describes and proves — with evidence generated
by the code, not hardcoded — that the data system is **correct, reproducible, auditable
and efficient**.

```
documents -> tokenized shards -> manifests -> mixture schedule -> packing
-> batches -> (fake) training -> consumption ledger -> learning ledger
-> checkpoint -> crash -> resume -> replay -> fork -> audit
```

## Run it (one command)

```bash
python run_demo.py          # numpy is the only third-party dependency
python -m unittest discover -s tests -p "test_*.py" -v   # the invariant tests
```

`run_demo.py` builds the corpus, freezes the tokenizer, shards + manifests it, compiles
the mixture schedule, runs training, **deliberately crashes, resumes, replays an interval,
and forks a branch**, audits everything, and writes `submission_artifacts/`. It exits 0
only if every requirement passes. No manual intervention.

## The one idea everything rests on

> **Every batch `B(t)` is a pure, reconstructable function of
> `(master_seed, compiled_schedule, shard_set, sampler_state_at_t)`.**

The sampler state (global step + per-lane cursor/epoch + deferral queue) is tiny and
fully serialisable, so a checkpoint captures the exact stream position. That is what makes
resume/replay/fork **provable** instead of asserted: restore the state, recompute `B(t)`,
and compare its content hash to the ledger. The data stream never depends on the model, the
wall clock, or a global RNG — so replay reconstructs batches with no model at all
(mirroring the lecture: on replay you *read and send* the recorded order, you don't
recalculate `[02:37:15]`).

## Architecture (`tdes/`)

| Module | Responsibility |
|---|---|
| `hashing.py` | canonical JSON + SHA-256; `derive_seed` makes per-step/per-lane RNGs a pure function of position |
| `tokenizer.py` | frozen byte-level BPE (self-contained, no network); identity = a content hash |
| `corpus.py` | tiny deterministic multi-lane corpus; structured (context/answer) docs + **eval and validation** splits |
| `shards.py` | immutable binary token shards + manifests (content hash, provenance, cleaning-pipeline hash, dedup/contamination/eval-overlap status); verified on load |
| `schedule.py` | curriculum stages, lane weights, **protected floors**; `lane_alloc(t)` is pure; feasibility check |
| `opus.py` | in-training selection — accept / reject / **defer**, scored on the token prefix; the stream applies the **protected-floor override** |
| `packing.py` | two policies by data type — **concat** (pretraining) and **structure-preserving** (SFT/agentic); loss mask, segment ids, **materialised causal block-diagonal attention mask**, position ids |
| `stream.py` | the deterministic batch stream; `SamplerState` + `next_batch(state)` |
| `model.py` | tiny bigram LM — real per-token cross-entropy; zero-init ⇒ initial loss = **ln(V)** (the lecture's −ln(1/vocab)) |
| `ledgers.py` | append-only, **hash-chained** consumption + learning ledgers; `offset` / `truncate_to` |
| `checkpoint.py` | model + RNG + sampler state + ledger offsets, content-hashed |
| `engine.py` | the training loop and the crash / resume / replay / fork operations |
| `audit.py` | recomputes every invariant from the artifacts → evidence bundle |
| `perf.py` | throughput + packing efficiency |

## Design decisions (and why)

- **Determinism by construction, not by luck.** The lecture warns a Python seed reproduces
  "only on that machine on that day" `[00:24:18]`. So reproducibility here rests on **recorded
  hashes + a reconstructable sampler state**, and the batch payload is **integer-only** (token
  ids + masks), which hashes identically across machines. Float math touches only the throwaway
  model, never a proof.
- **Protected floors override OPUS, in the stream.** OPUS scoring is a pure function; the
  *floor* logic lives in the stream because only it knows batch composition. Protected lanes
  (Indic/agentic/reasoning) are never dropped by OPUS — every override is logged.
- **Two packing policies, by data type** `[02:24:06]`: pretraining lanes concat-and-chop for
  high utilisation; SFT/agentic lanes are structure-preserving (one trajectory per sequence,
  loss only on answer spans, tool observations masked).
- **Checkpoints tie to ledger offsets.** Resume truncates the ledger to the checkpoint offset
  and continues, so no batch is skipped or repeated.
- **Evidence is generated.** `audit.py` recomputes hashes, re-derives batches, scans for eval
  leakage, and checks mixture compliance — then writes `evidence.json/md`. Nothing is asserted
  by hand.

## What the demo proves (maps to the grading areas)

| Area | Proof in the run |
|---|---|
| Shards / manifests / tokenizer | every shard's content + tokenizer hash re-verifies; tamper is caught (test) |
| Packing / masks / batch | no loss across a segment boundary or onto a pad target; position ids reset per doc; the **attention mask** is causal, block-diagonal, and never touches pad (262,144 pairs checked) |
| Mixture / floors / OPUS | planned vs actual lane shares within tolerance; floors always met; OPUS accept/reject/defer with **protected-floor overrides** on Indic/agentic/reasoning |
| Consumption + learning ledgers | hash-chained; loss linked to source shard/doc; loss falls from **ln(V)** |
| Checkpoint / crash / resume / replay / fork | crash at step 25 → resume from ckpt 20 → **the full resumed stream is hash-identical** to a clean run; replay matches **batch ids, token spans and hashes**; a fork with a new seed **diverges** yet stays self-consistent |
| Evaluation & validation firewall | both held-out splits withheld; a scan of every consumed document instance confirms **no held-out document** enters any loss-bearing batch |
| Throughput / packing efficiency | batches/s, useful-loss-bearing tokens/s, packing utilisation — measured, not stated |

## `submission_artifacts/` (generated)

```
submission_artifacts/
  run.log              # the full event sequence with [PASS] markers
  evidence.json        # every requirement PASS/FAIL + where the evidence lives
  evidence.md          # the human-readable summary table
  performance.json     # throughput + packing efficiency
  manifests/           # one per shard + index.json
  ledgers/             # consumption.jsonl, learning.jsonl, opus_trail.jsonl
  checkpoints/         # ckpt_*.json metadata (weight *.bin regenerate on run — gitignored)
```

## The completion bar

> *"The assignment is complete only when the system can prove what it consumed, why it consumed it, what the
> model learned from it and how the run can be reconstructed."*

| Must prove | Mechanism | Evidence |
|---|---|---|
| **what it consumed** | consumption ledger, append-only + hash-chained | 40 entries, chain verified, 1,962 consumed document instances |
| **why it consumed it** | mixture-schedule allocation + OPUS decision record (score, reason, override flag), joined to the ledger by `doc_id` | 2,245 decisions; a test asserts every consumed doc has a decision record |
| **what the model learned** | learning ledger — per-sequence *and* per-token loss, each linked to its source shard/document | loss 6.4907 → 6.4167, starting exactly at ln(V) |
| **how it can be reconstructed** | checkpoint → resume → replay → fork, each verified by hash | resumed stream identical; replay matches batch ids + 295 token spans + hashes |

Scope note (per the brief `[02:26:07]`): the real service, dashboard, UI, and throughput
engineering are out of scope; the model is deliberately trivial ("training is just fake
training" `[02:23:06]`). The point is the **data system**, and its correctness is reproducible.
