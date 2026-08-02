#!/usr/bin/env python3
"""ERA V5 · Session 6 — one command that runs the complete Training Data Execution
System demonstration and writes submission_artifacts/.

    python run_demo.py

Path exercised end to end:
  documents -> tokenized shards -> manifests -> mixture schedule -> packing
  -> batches -> (fake) training -> consumption + learning ledgers -> checkpoint
  -> crash -> resume -> replay -> fork -> audit -> evidence.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import time

from tdes.audit import (build_evidence, check_packing, evidence_markdown,
                        firewall_scan, mixture_compliance)
from tdes.corpus import HELD_OUT_SPLITS, build_corpus, corpus_stats
from tdes.engine import CrashSignal, TrainingEngine
from tdes.hashing import canonical
from tdes.model import BigramLM
from tdes.opus import summarize as opus_summary
from tdes.perf import performance_report
from tdes.schedule import DEFAULT_STAGES, compile_schedule, feasibility
from tdes.shards import ShardSet
from tdes.stream import SamplerState
from tdes.tokenizer import FrozenTokenizer

MASTER_SEED = 6006
BATCH_SIZE = 16
NUM_MERGES = 400
TOTAL_STEPS = 40
CHECKPOINT_EVERY = 10
CRASH_AT = 25
RESUME_STEP = 20            # last checkpoint <= CRASH_AT
REPLAY = (12, 18)
FORK_FROM, FORK_SEED, FORK_STEPS = 20, 99, 5

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "submission_artifacts"


class Log:
    def __init__(self, path):
        self.lines = []
        self.path = path

    def __call__(self, msg):
        line = msg if msg.startswith("[") or msg[:1].isupper() else msg
        print(line)
        self.lines.append(line)

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")


def main():
    if ART.exists():
        shutil.rmtree(ART)
    ART.mkdir(parents=True)
    log = Log(ART / "run.log")
    log(f"=== TDES demo — master_seed={MASTER_SEED} batch_size={BATCH_SIZE} "
        f"steps={TOTAL_STEPS} ===")

    # 1) corpus + frozen tokenizer -------------------------------------------
    docs = build_corpus()
    stats = corpus_stats(docs)
    log(f"corpus built: {stats['n_docs']} docs, lanes={stats['by_lane']}, "
        f"split={stats['by_split']}")
    tok = FrozenTokenizer.train([d["text"] for d in docs], NUM_MERGES)
    log(f"tokenizer frozen: vocab={tok.vocab_size} hash={tok.tokenizer_hash[:16]}")

    # 2) shards + manifests ---------------------------------------------------
    shardset = ShardSet.build(docs, tok)
    index = shardset.write(ART)
    log(f"shards created: {len(shardset.shards)} shards "
        f"({sum(s.manifest['n_tokens'] for s in shardset.shards)} tokens)")
    verify = shardset.verify()
    if all(r["tokenizer_hash_ok"] for r in verify):
        log("[PASS] tokenizer_hash_verified")
    if all(r["content_hash_ok"] for r in verify):
        log(f"manifests validated: {len(verify)} shards, content hashes OK")

    # 3) evaluation + validation firewalls ------------------------------------
    held_out_shards = shardset.held_out_shards()
    trainable_ids = {s.shard_id for s in shardset.trainable()}
    blocked = [s.shard_id for s in held_out_shards if s.shard_id not in trainable_ids]
    for sid in blocked:
        log(f"evaluation data blocked: shard {sid} withheld from training")
    held_out = {sp: {d["doc_id"] for d in docs if d["split"] == sp}
                for sp in HELD_OUT_SPLITS}
    blocked_ok = len(blocked) == len(held_out_shards) and len(blocked) > 0
    if blocked_ok:
        n_eval = sum(1 for s in held_out_shards if s.split == "eval")
        n_val = len(held_out_shards) - n_eval
        log(f"[PASS] eval_shard_blocked ({n_eval} eval + {n_val} validation shards "
            f"withheld, {len(blocked)} total)")

    # 4) mixture schedule -----------------------------------------------------
    schedule = compile_schedule(DEFAULT_STAGES, MASTER_SEED, BATCH_SIZE)
    tokens_per_lane = {}
    for s in shardset.trainable():
        tokens_per_lane[s.lane] = tokens_per_lane.get(s.lane, 0) + s.manifest["n_tokens"]
    feas = feasibility(schedule, tokens_per_lane)
    log(f"mixture compiled: {len(schedule.stages)} stages, {schedule.total_steps} steps, "
        f"feasible={feas['feasible']}")

    # 5) reference (clean) run — the 'expected' ground truth ------------------
    ref = TrainingEngine(shardset, schedule, tok, ART / "_ref", MASTER_SEED, "ref")
    ref.run(TOTAL_STEPS, CHECKPOINT_EVERY)
    ref_hashes = dict(ref.batch_hashes)
    log(f"reference run complete: {len(ref_hashes)} batches (expected stream)")

    # 6) main run: train, checkpoint, CRASH, resume, finish -------------------
    eng = TrainingEngine(shardset, schedule, tok, ART, MASTER_SEED, "main")
    try:
        eng.run(TOTAL_STEPS, CHECKPOINT_EVERY, crash_at=CRASH_AT, log=log)
    except CrashSignal as e:
        log(f"crash simulated: {e}")
    log(f"batches packed so far: {len(eng.batch_hashes)}; "
        f"OPUS decisions recorded: {len(eng.opus_trail)}")
    t_resume0 = time.perf_counter()
    eng.resume_from(RESUME_STEP, log=log)
    resume_latency = time.perf_counter() - t_resume0
    eng.run(TOTAL_STEPS - RESUME_STEP, CHECKPOINT_EVERY, log=log)

    # prove: the whole resumed stream matches the expected stream (no skip/repeat)
    resume_next_expected = ref_hashes[RESUME_STEP]
    resume_next_actual = eng.batch_hashes[RESUME_STEP]
    all_match = all(eng.batch_hashes[t] == ref_hashes[t] for t in ref_hashes)
    resume_ok = all_match and resume_next_expected == resume_next_actual
    if resume_ok:
        log(f"[PASS] resume_next_batch_matched step={RESUME_STEP} "
            f"sha={resume_next_actual[:12]} (full stream identical: {all_match})")
    else:
        log("[FAIL] resume mismatch")

    # 7) replay ---------------------------------------------------------------
    t_replay0 = time.perf_counter()
    replay = eng.replay_interval(*REPLAY)
    replay_latency = time.perf_counter() - t_replay0
    log(f"historical stream replayed: interval {REPLAY} match={replay['all_match']}")
    if replay["all_match"]:
        log(f"[PASS] replay_hash_matched interval={list(REPLAY)}")

    # 8) fork -----------------------------------------------------------------
    fork = eng.fork_from(FORK_FROM, FORK_SEED, FORK_STEPS, log=None)
    log(f"branch forked: from step {FORK_FROM} seed {FORK_SEED} "
        f"diverged={fork['diverged']} self_consistent={fork['self_consistent']}")

    # 9) audit ----------------------------------------------------------------
    # reconstruct a real consumed batch (anneal step, has agentic) for packing check
    entry30 = eng.consumption.find_step(30)
    st30 = SamplerState.from_dict(entry30["state_before"])
    batch30, _, _, _ = eng.stream.next_batch(st30)
    packing = check_packing(batch30, tok.pad_id)
    mixture = mixture_compliance(eng.consumption.entries, schedule)
    firewall = firewall_scan(eng.consumption.entries, held_out)

    avg_util = sum(p["real_tokens"] for p in eng.perf) / max(
        1, sum(p["n_seq"] for p in eng.perf) * schedule.stages[0]["seq_len"])
    perf = performance_report(eng.perf, {
        "avg_util": avg_util,
        "capacity_tokens": sum(p["n_seq"] for p in eng.perf) * schedule.stages[0]["seq_len"],
        "resume_latency_sec": resume_latency, "replay_latency_sec": replay_latency})
    log(f"performance measured: {perf['batches_per_sec']} batches/s, "
        f"{perf['useful_loss_bearing_tokens_per_sec']} useful-loss-tok/s, "
        f"util={perf['avg_packing_utilization']}")

    loss_start = eng.learning.entries[0]["mean_loss"]
    loss_end = eng.learning.entries[-1]["mean_loss"]
    ctx = {
        "shard_verify": verify, "firewall": firewall, "blocked_ok": blocked_ok,
        "n_eval_blocked": len(blocked), "packing": packing, "mixture": mixture,
        "opus_summary": opus_summary(eng.opus_trail),
        "consumption_chain_ok": eng.consumption.verify_chain(),
        "learning_chain_ok": eng.learning.verify_chain(),
        "consumption_len": eng.consumption.offset(),
        "loss_decreased": loss_end < loss_start,
        "loss_start": loss_start, "loss_end": loss_end,
        "loss_theory": round(BigramLM.initial_loss(tok.vocab_size), 3),
        "resume_ok": resume_ok, "resume_step": RESUME_STEP,
        "resume_next_expected": resume_next_expected,
        "resume_next_actual": resume_next_actual,
        "replay": replay, "fork": {k: v for k, v in fork.items() if k != "engine"},
        "performance": perf, "e2e_ok": True, "total_steps": TOTAL_STEPS,
    }
    evidence = build_evidence(ctx)
    log(f"audit completed: overall={evidence['overall']}")

    # 10) write artifacts -----------------------------------------------------
    (ART / "ledgers").mkdir(exist_ok=True)
    eng.consumption.write(ART / "ledgers" / "consumption.jsonl")
    eng.learning.write(ART / "ledgers" / "learning.jsonl")
    with (ART / "ledgers" / "opus_trail.jsonl").open("wb") as f:
        for r in eng.opus_trail:
            f.write(canonical(r) + b"\n")
    (ART / "performance.json").write_bytes(canonical(perf))
    (ART / "evidence.json").write_bytes(json.dumps(evidence, indent=2).encode())
    (ART / "evidence.md").write_text(evidence_markdown(evidence))
    shutil.rmtree(ART / "_ref", ignore_errors=True)     # reference run is scaffolding
    log.flush()

    print("\n" + "=" * 68)
    print(f"OVERALL: {evidence['overall']}  ·  artifacts in {ART}")
    for r in evidence["requirements"]:
        print(f"  [{r['result']}] {r['area']:<34} {r['requirement']}")
    print("=" * 68)
    return 0 if evidence["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
