"""TDES — Training Data Execution System (ERA V5 · Session 6).

A small but complete, fully-deterministic data pipeline for LLM training:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing
    -> batches -> (fake) training -> consumption ledger -> learning ledger
    -> checkpoint -> crash -> resume -> replay -> audit

The one invariant everything rests on:

    every batch B(t) is a PURE, RECONSTRUCTABLE function of
        (master_seed, compiled_schedule, shard_set, sampler_state_at_t)

The sampler state is small and fully serialisable, so a checkpoint captures the
exact stream position. That is what makes resume/replay/fork *provable* instead
of asserted: recompute B(t) and compare its content hash to the ledger.
"""

__version__ = "1.0.0"
