#!/usr/bin/env python3
"""Train the V5-class tokenizer the collision census runs against.

The census is only meaningful on a *real* vocabulary: the whole question is whether
real Indic subword tokens exceed a 32-byte window, and synthetic strings cannot answer
that. We reuse the Session 3 India-first corpus (43 MB, 13 languages + code + math)
and the same BPE recipe (NFKC + Metaspace), at the V5 vocabulary size of 131,072.

    python -m dyn.build_vocab            # writes artifacts/tokenizer-131072.json

Idempotent: skips if the tokenizer already exists.
"""
import pathlib
import sys
import time

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Session 3 already assembled an Indic-heavy corpus; reuse it rather than re-fetching.
CORPUS = ROOT.parent / "session-03-india-first-40b" / "data" / "train"
VOCAB_SIZE = 131_072                      # the V5 number, 2^17
OUT = ROOT / "artifacts" / f"tokenizer-{VOCAB_SIZE}.json"


def make_tokenizer() -> Tokenizer:
    """Same recipe as the Session 3 sweep, so fertility numbers stay comparable."""
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.normalizer = normalizers.NFKC()
    t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
    t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
    return t


def main() -> int:
    if OUT.exists():
        print(f"cached: {OUT.relative_to(ROOT)}")
        return 0
    files = sorted(str(p) for p in CORPUS.glob("*.txt"))
    if not files:
        print(f"ERROR: no corpus at {CORPUS}", file=sys.stderr)
        print("The corpus is large and third-party, so it is not committed. Fetch it with:\n"
              "    cd ../session-03-india-first-40b && python fetch_corpus.py",
              file=sys.stderr)
        return 1

    mb = sum(pathlib.Path(f).stat().st_size for f in files) / 1e6
    print(f"training BPE vocab={VOCAB_SIZE:,} on {len(files)} files ({mb:.0f} MB)")
    t0 = time.time()
    tok = make_tokenizer()
    tok.train(files, BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,
                                special_tokens=["[UNK]"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT))
    print(f"done in {time.time()-t0:.0f}s -> {OUT.relative_to(ROOT)} "
          f"(actual vocab {tok.get_vocab_size():,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
