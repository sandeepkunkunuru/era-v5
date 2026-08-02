"""Throughput + packing efficiency (`[02:22:06]`): raw vs useful loss-bearing tokens,
packing utilisation, and latencies. Measured from the actual run, never hardcoded."""
from __future__ import annotations

from typing import Dict, List


def performance_report(perf_rows: List[dict], extra: Dict) -> dict:
    steps = len(perf_rows)
    real = sum(p["real_tokens"] for p in perf_rows)
    loss = sum(p["loss_tokens"] for p in perf_rows)
    secs = sum(p["seconds"] for p in perf_rows) or 1e-9
    cap = sum(p["n_seq"] for p in perf_rows)
    return {
        "steps": steps,
        "wall_seconds_stream_train": round(secs, 4),
        "batches_per_sec": round(steps / secs, 2),
        "real_tokens_per_sec": round(real / secs, 1),
        "useful_loss_bearing_tokens_per_sec": round(loss / secs, 1),
        "total_real_tokens": real,
        "total_loss_bearing_tokens": loss,
        "avg_packing_utilization": round(extra.get("avg_util", 0.0), 4),
        "avg_loss_bearing_fraction": round(loss / (extra.get("capacity_tokens", 1) or 1), 4),
        "resume_latency_sec": round(extra.get("resume_latency_sec", 0.0), 4),
        "replay_latency_sec": round(extra.get("replay_latency_sec", 0.0), 4),
    }
