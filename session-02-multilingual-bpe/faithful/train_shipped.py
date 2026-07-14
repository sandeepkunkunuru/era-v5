#!/usr/bin/env python3
"""Train the SHIPPED tokenizer: parity-aware BPE on the faithful base (en/hi/te/mai).

Same faithful setup as the reference (NFKC + Metaspace pre-tok/decoder + every corpus character in
the vocab, so decode(encode(text)) is lossless), but the merges are chosen the parity-aware way
(Foroutan et al., ACL 2026) instead of by fixed training weights:

    at every merge step, take the most-frequent adjacent pair of the language with the CURRENTLY
    HIGHEST fertility (tokens / faithful_units), and add it to the shared vocabulary.

This directly minimizes the spread the score measures — `score = 1000 / (max_fertility - min_fertility)`.
The assignment's 1.2 cap never binds (all four languages sit near 0.61), so we run pure parity: all
four fertilities collapse to ~0.610 and the spread falls to ~6e-5. The result is a single ordinary
HuggingFace BPE tokenizer (vocab + ordered merges), re-runnable by the grader, and it is measured
here with the real `tokenizers` library — not the training simulation.

Baseline for comparison (fixed-weight reference, score 6502.56): reproduce_reference.py.

    python train_shipped.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import regex
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT_TOKENIZER = ROOT / "tokenizer.json"
OUT_METRICS = ROOT / "metrics.json"

LANGS = ["en", "hi", "te", "mai"]
VOCAB = 10000
FU_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
NFKC = normalizers.NFKC()


def metaspace_pieces(text: str) -> list[str]:
    """NFKC then Metaspace(▁, prepend=never): space→▁, each ▁ starts a new piece; \\t \\n stay literal."""
    s = NFKC.normalize_str(text).replace(" ", "▁")
    pieces, cur = [], ""
    for c in s:
        if c == "▁":
            if cur:
                pieces.append(cur)
            cur = "▁"
        else:
            cur += c
    if cur:
        pieces.append(cur)
    return pieces


RAW = {l: (CORPUS / f"{l}.faithful.txt").read_text(encoding="utf-8") for l in LANGS}
UNITS = {l: len(FU_RE.findall(RAW[l])) for l in LANGS}
PIECES = {l: Counter(metaspace_pieces(RAW[l])) for l in LANGS}


def train_parity() -> tuple[set, list]:
    reps = {l: {w: list(w) for w in PIECES[l]} for l in LANGS}
    tok_total = {l: sum(len(reps[l][w]) * PIECES[l][w] for w in reps[l]) for l in LANGS}
    paircnt = {l: Counter() for l in LANGS}
    pairloc = {l: defaultdict(set) for l in LANGS}
    pairs = lambda s: [(s[i], s[i + 1]) for i in range(len(s) - 1)]
    for l in LANGS:
        for w, sym in reps[l].items():
            c = PIECES[l][w]
            for p in pairs(sym):
                paircnt[l][p] += c
                pairloc[l][p].add(w)
    vocab = set(ch for l in LANGS for w in PIECES[l] for ch in w) | {"[UNK]"}
    merges = []
    fert = lambda l: tok_total[l] / UNITS[l]

    def apply_merge(l, a, b):
        ab = a + b
        for w in list(pairloc[l].get((a, b), ())):
            sym = reps[l][w]
            c = PIECES[l][w]
            for p in pairs(sym):
                paircnt[l][p] -= c
                if paircnt[l][p] <= 0:
                    paircnt[l].pop(p, None)
                pairloc[l][p].discard(w)
            out, i, removed = [], 0, 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(ab); i += 2; removed += 1
                else:
                    out.append(sym[i]); i += 1
            reps[l][w] = out
            tok_total[l] -= removed * c
            for p in pairs(out):
                paircnt[l][p] += c
                pairloc[l][p].add(w)

    while len(vocab) < VOCAB:
        cand = [l for l in LANGS if paircnt[l]]
        if not cand:
            break
        L = max(cand, key=fert)                      # worst-compressed language (ties -> LANGS order)
        # its most valuable pair; DETERMINISTIC tie-break (freq desc, then pair lexicographically)
        # so the run is reproducible regardless of PYTHONHASHSEED / set-iteration order.
        (a, b) = max(paircnt[L].items(), key=lambda kv: (kv[1], kv[0]))[0]
        for l in LANGS:
            if (a, b) in pairloc[l]:
                apply_merge(l, a, b)
        merges.append((a, b))
        vocab.add(a + b)
    return vocab, merges


def build(vocab, merges) -> Tokenizer:
    uv = {"[UNK]": 0}
    for t in sorted(vocab - {"[UNK]"}):
        uv.setdefault(t, len(uv))
    tok = Tokenizer(models.BPE(vocab=uv, merges=[tuple(m) for m in merges], unk_token="[UNK]"))
    tok.normalizer = NFKC
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
    return tok


def main() -> int:
    vocab, merges = train_parity()
    tok = build(vocab, merges)
    token_counts = {l: len(tok.encode(RAW[l]).ids) for l in LANGS}   # authoritative HF counts
    ratios = {l: token_counts[l] / UNITS[l] for l in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    metrics = {
        "variant": "wiki_faithful_markdown",
        "method": "parity-aware BPE (Foroutan et al. ACL 2026): each step merges the worst-compressed language's pair",
        "languages": {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"},
        "vocab_size": tok.get_vocab_size(),
        "faithful_units": UNITS,
        "unit_policy": "contiguous L/M/N run = 1 unit; each visible non-space punctuation/symbol = 1 unit",
        "token_counts": token_counts,
        "ratios": ratios,
        "spread": spread,
        "score": 1000 / spread,
        "baseline_reference_score": 6502.558365188569,
    }
    tok.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
