"""Build a 50M-token byte-level corpus from the Session 4 cleaned shard.

Session 4 cleaned a 69.4M-token slice of OpenThoughts-114k and wrote it to a parquet shard. That is
the corpus this session trains on, so the data path is the one this course actually built rather
than another copy of tiny Shakespeare. Tokens here are raw bytes (vocab 256): the tokenizer question
belongs to Session 2, and a byte vocabulary keeps the embedding out of the parameter budget so the
model is 21M parameters of transformer rather than 21M parameters of lookup table.

    python data.py          # writes train.bin (50M bytes) and val.bin (2M bytes)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHARD = HERE.parent / "session-04-data-cleaning/data/shard0.parquet"
TRAIN_TOKENS = 50_000_000
VAL_TOKENS = 2_000_000


def build():
    import pyarrow.parquet as pq

    if not SHARD.exists():
        sys.exit(f"missing {SHARD} -- run Session 4's pipeline first, or point SHARD elsewhere")
    need = TRAIN_TOKENS + VAL_TOKENS
    buf, total = [], 0
    f = pq.ParquetFile(SHARD)
    for batch in f.iter_batches(batch_size=64, columns=["conversations"]):
        for convo in batch.to_pylist()["conversations"] if isinstance(batch.to_pylist(), dict) else [r["conversations"] for r in batch.to_pylist()]:
            for turn in convo:
                b = turn["value"].encode("utf-8", errors="ignore") + b"\n"
                buf.append(b)
                total += len(b)
        if total >= need:
            break
    if total < need:
        sys.exit(f"shard held only {total:,} bytes, need {need:,}")
    data = np.frombuffer(b"".join(buf)[:need], dtype=np.uint8)
    data[:TRAIN_TOKENS].tofile(HERE / "train.bin")
    data[TRAIN_TOKENS:].tofile(HERE / "val.bin")
    print(f"train.bin {TRAIN_TOKENS:,} tokens · val.bin {VAL_TOKENS:,} tokens · vocab 256 (bytes)")


def load(split):
    p = HERE / f"{split}.bin"
    if not p.exists():
        build()
    return np.memmap(p, dtype=np.uint8, mode="r")


if __name__ == "__main__":
    build()
