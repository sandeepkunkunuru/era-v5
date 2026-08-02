"""The training engine: drives the stream + model + ledgers + checkpoints, and
implements the crash / resume / replay / fork operations whose correctness is the
whole point of the assignment.

Every step: pull B(t) from the stream (pure fn of sampler state), fake-train the
model on it, write the consumption + learning ledger entries, advance the RNG, and
checkpoint on cadence. A checkpoint is tied to the ledger offset, so resume can
truncate and continue with no skipped or repeated batch.
"""
from __future__ import annotations

import pathlib
import time
from typing import Dict, List, Optional

import numpy as np

from . import checkpoint as ckpt
from .ledgers import Ledger
from .model import BigramLM
from .opus import Opus
from .schedule import Schedule
from .shards import ShardSet
from .stream import SamplerState, Stream


class CrashSignal(Exception):
    """Raised to simulate a mid-run crash (`[01:45:24]` — the imprecise kill)."""


class TrainingEngine:
    def __init__(self, shards: ShardSet, schedule: Schedule, tokenizer,
                 workdir: pathlib.Path, master_seed: int, run_name: str = "main"):
        self.shards = shards
        self.schedule = schedule
        self.tok = tokenizer
        self.workdir = pathlib.Path(workdir)
        self.master_seed = master_seed
        self.run_name = run_name
        self.opus = Opus(master_seed)
        self.stream = Stream(shards, schedule, self.opus,
                             tokenizer.eos_id, tokenizer.pad_id, master_seed)
        self.model = BigramLM(tokenizer.vocab_size)
        self.rng = np.random.default_rng(master_seed)
        self.sampler = self.stream.initial_state()
        self.consumption = Ledger("consumption")
        self.learning = Ledger("learning")
        self.opus_trail: List[dict] = []
        self.checkpoints: List[dict] = []
        self.batch_hashes: Dict[int, str] = {}
        self.perf: List[dict] = []

    # ---- learning-ledger record --------------------------------------------
    def _learning_record(self, batch: dict, out: dict) -> dict:
        seqs = batch  # batch carries stacked arrays; we need per-seq slices
        per_seq = []
        pos_loss = out.get("pos_loss")
        mask = out.get("loss_mask_used")
        tokens = batch["tokens"]
        seq_meta_source = batch.get("_seq_meta", [])
        token_samples = []
        if pos_loss is not None:
            for b in range(pos_loss.shape[0]):
                mrow = mask[b]
                n = int(mrow.sum())
                mean = float(pos_loss[b][mrow].mean()) if n else 0.0
                meta = seq_meta_source[b] if b < len(seq_meta_source) else {}
                first = meta.get("members", [{}])[0] if meta.get("members") else {}
                per_seq.append({"seq_index": b, "lane": meta.get("lane"),
                                "mean_loss": round(mean, 5), "n_loss_tokens": n,
                                "source_shard": first.get("shard_id"),
                                "source_doc": first.get("doc_id")})
            # token-level: the few highest-loss targets, linked to source
            flat = []
            for b in range(pos_loss.shape[0]):
                idxs = np.where(mask[b])[0]
                for i in idxs:
                    flat.append((float(pos_loss[b][i]), b, int(i)))
            flat.sort(reverse=True)
            for loss_v, b, i in flat[:5]:
                tgt = int(tokens[b][i + 1])
                meta = seq_meta_source[b] if b < len(seq_meta_source) else {}
                first = meta.get("members", [{}])[0] if meta.get("members") else {}
                token_samples.append({
                    "seq_index": b, "position": i, "target_token_id": tgt,
                    "token_preview": self.tok.decode([tgt])[:24],
                    "loss": round(loss_v, 5), "lane": meta.get("lane"),
                    "source_shard": first.get("shard_id"),
                    "source_doc": first.get("doc_id")})
        return {"step": batch["step"], "batch_id": batch["batch_id"],
                "mean_loss": round(out["mean_loss"], 5),
                "n_loss_tokens": out["n_loss_tokens"],
                "weight_hash_after": self.model.weight_hash()[:16],
                "per_sequence": per_seq, "token_level_samples": token_samples}

    # ---- main loop ----------------------------------------------------------
    def run(self, steps: int, checkpoint_every: int, crash_at: Optional[int] = None,
            log=None) -> Dict[int, str]:
        target = self.sampler.t + steps
        while self.sampler.t < target:
            t = self.sampler.t
            tic = time.perf_counter()
            batch, new_state, crec, opus_recs = self.stream.next_batch(self.sampler)
            batch["_seq_meta"] = crec["sequences"]
            out = self.model.step(batch["tokens"], batch["loss_mask"])
            dt = time.perf_counter() - tic
            # ledgers
            crec["opus_decisions"] = len(opus_recs)
            crec["weight_hash_before"] = None
            self.consumption.append(crec)
            self.learning.append(self._learning_record(batch, out))
            self.opus_trail.extend(opus_recs)
            self.batch_hashes[t] = batch["batch_sha256"]
            self.perf.append({"step": t, "n_seq": batch["tokens"].shape[0],
                              "real_tokens": batch["utilization"]["real_tokens"],
                              "loss_tokens": batch["utilization"]["loss_tokens"],
                              "seconds": dt})
            # advance RNG (demonstrates RNG must be checkpointed)
            _ = self.rng.random()
            self.sampler = new_state
            if log:
                log(f"step {t:03d} stage={batch['stage']:<7} batch={batch['batch_id']} "
                    f"loss={out['mean_loss']:.3f} util={batch['utilization']['packing_utilization']:.2f} "
                    f"sha={batch['batch_sha256'][:12]}")
            if (t + 1) % checkpoint_every == 0:
                self._checkpoint(t + 1, log)
            if crash_at is not None and self.sampler.t == crash_at:
                raise CrashSignal(f"simulated crash at step {crash_at}")
        return self.batch_hashes

    def _checkpoint(self, step: int, log=None) -> dict:
        meta = ckpt.save(self.workdir / "checkpoints", step, self.model, self.rng,
                         self.sampler, self.consumption.offset(),
                         self.learning.offset(), self.tok.tokenizer_hash,
                         self.master_seed)
        self.checkpoints.append(meta)
        if log:
            log(f"[PASS] checkpoint_saved step={step} id={meta['checkpoint_id'][:12]} "
                f"cons_off={meta['consumption_offset']} learn_off={meta['learning_offset']}")
        return meta

    # ---- resume -------------------------------------------------------------
    def resume_from(self, step: int, log=None) -> dict:
        meta = ckpt.load(self.workdir / "checkpoints", step, self.model, self.rng)
        self.sampler = ckpt.sampler_from(meta)
        dropped_c = self.consumption.truncate_to(meta["consumption_offset"])
        dropped_l = self.learning.truncate_to(meta["learning_offset"])
        # drop batch-hash records after the checkpoint too
        self.batch_hashes = {k: v for k, v in self.batch_hashes.items() if k < step}
        self.perf = [p for p in self.perf if p["step"] < step]
        if log:
            log(f"run resumed from checkpoint step={step} "
                f"(dropped {dropped_c} consumption + {dropped_l} learning entries)")
        return meta

    # ---- replay -------------------------------------------------------------
    @staticmethod
    def _token_spans(sequences: List[dict]) -> list:
        """The (shard, doc, token-count) spans that make up each packed sequence —
        the 'token spans' the spec requires replay to match, independent of hashes."""
        return [[[m["shard_id"], m["doc_id"], m["n_tokens"]] for m in s["members"]]
                for s in sequences]

    def replay_interval(self, t0: int, t1: int) -> dict:
        """Reconstruct batches for [t0, t1) from the recorded state_before at t0 and
        prove that **batch ids, token spans and hashes** all match the original run.
        No model is involved — the data stream does not depend on the weights."""
        entry = self.consumption.find_step(t0)
        state = SamplerState.from_dict(entry["state_before"])
        results = []
        ok = True
        for t in range(t0, t1):
            batch, state, crec, _ = self.stream.next_batch(state)
            rec = self.consumption.find_step(t)
            id_match = batch["batch_id"] == rec["batch_id"]
            hash_match = batch["batch_sha256"] == rec["batch_sha256"]
            spans_new = self._token_spans(crec["sequences"])
            spans_old = self._token_spans(rec["sequences"])
            span_match = spans_new == spans_old
            n_spans = sum(len(s) for s in spans_new)
            ok = ok and id_match and hash_match and span_match
            results.append({"step": t,
                            "batch_id": rec["batch_id"], "batch_id_match": id_match,
                            "recomputed": batch["batch_sha256"][:12],
                            "recorded": rec["batch_sha256"][:12],
                            "hash_match": hash_match,
                            "token_spans_compared": n_spans,
                            "token_spans_match": span_match,
                            "match": id_match and hash_match and span_match})
        return {"interval": [t0, t1], "all_match": ok,
                "proved": ["batch_id", "token_spans", "batch_sha256"],
                "total_token_spans_compared": sum(r["token_spans_compared"]
                                                  for r in results),
                "steps": results}

    # ---- fork ---------------------------------------------------------------
    def fork_from(self, step: int, new_seed: int, n_steps: int, log=None) -> dict:
        """Branch a new run from a checkpoint with a different seed -> divergent but
        valid stream. Proves the fork differs from the original at the same steps."""
        forked = TrainingEngine(self.shards, self.schedule, self.tok,
                                self.workdir / "fork", new_seed, run_name="fork")
        meta = ckpt.load(self.workdir / "checkpoints", step, forked.model, forked.rng)
        forked.sampler = ckpt.sampler_from(meta)
        forked.sampler.t = step
        forked.run(n_steps, checkpoint_every=n_steps + 1, log=log)
        comparisons = []
        diverged = False
        for t in range(step, step + n_steps):
            orig = self.batch_hashes.get(t)
            fk = forked.batch_hashes.get(t)
            differ = (orig is not None and fk is not None and orig != fk)
            diverged = diverged or differ
            comparisons.append({"step": t, "original": (orig or "")[:12],
                                "fork": (fk or "")[:12], "differs": differ})
        return {"fork_seed": new_seed, "from_step": step, "diverged": diverged,
                "self_consistent": forked.consumption.verify_chain(),
                "steps": comparisons, "engine": forked}
