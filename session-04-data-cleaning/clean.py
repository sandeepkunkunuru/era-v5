#!/usr/bin/env python3
"""ERA V5 · Session 4 — the 8-strategy data-cleaning pipeline, run for real.

Applies the eight cleaning stages taught in the Session 4 lecture to a 10-100M
token slice of a reasoning-distillation dataset (OpenThoughts-114k), and records
exactly what each stage removed and why. Every number in the widget comes from
this script; nothing is asserted.

    stage 1  extract        strip repeated boilerplate + ghost markers
    stage 2  normalize      unicode NFC, entities, zero-width/control chars
    stage 3  language-ID    keep target languages; drop mislabeled/off-target
    stage 4  quality        Gopher/C4 heuristics + reasoning-specific defects
    stage 5  dedup          MinHash + LSH near-duplicate removal (from scratch)
    stage 6  PII scrub      redact emails / phones / IPs / handles (keep names)
    stage 7  decontaminate  n-gram overlap vs GSM8K / MATH-500 / HumanEval tests
    stage 8  manifest       per-shard provenance + reproducibility record

Deterministic: same input + same code -> same output (fixed MinHash seeds,
sorted iteration). Run:  python clean.py [N_ROWS]
"""
from __future__ import annotations
import hashlib, html, json, re, sys, unicodedata, time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import regex as re2  # unicode-property aware
import tiktoken

HERE = Path(__file__).parent
DATA = HERE / "data"
REPORT = HERE / "report"
REPORT.mkdir(exist_ok=True)

SHARD = DATA / "shard0.parquet"
SOURCE_REPO = "open-thoughts/OpenThoughts-114k"
SOURCE_COMMIT = "bd093c3994fd54d2390985b66988ddf282a55eb6"  # resolve hash seen on download
SOURCE_LICENSE = "apache-2.0"
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

ENC = tiktoken.get_encoding("cl100k_base")
def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))

# Ghost markers: OpenThoughts wraps every trace in these scaffolding tags. They
# are structure, not training signal — the exact "ghost marker" case from lecture.
MARKERS = ["<|begin_of_thought|>", "<|end_of_thought|>",
           "<|begin_of_solution|>", "<|end_of_solution|>"]

# --- provenance for reproducibility (stage 8) ---
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

SCRIPT_HASH = sha256_file(Path(__file__))
EXAMPLES = {}  # stage -> {before, after, note} for the widget

def log(msg): print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
def load_docs(n):
    t = pq.read_table(SHARD)
    systems = t.column("system").to_pylist()[:n]
    convs = t.column("conversations").to_pylist()[:n]
    docs = []
    for i, (sysprompt, conv) in enumerate(zip(systems, convs)):
        user = next((c["value"] for c in conv if c.get("from") == "user"), "")
        asst = next((c["value"] for c in conv if c.get("from") == "assistant"), "")
        docs.append({"id": i, "system": sysprompt or "", "user": user, "asst": asst,
                     "text": "", "tokens": 0, "flags": []})
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — EXTRACT: drop boilerplate + ghost markers, keep the training signal
# ─────────────────────────────────────────────────────────────────────────────
def stage_extract(docs, stats):
    # The `system` prompt is identical on (almost) every row — the "boilerplate on
    # every page" case. Measure how identical, then drop it from the training text.
    sys_hashes = defaultdict(int)
    for d in docs:
        sys_hashes[hashlib.md5(d["system"].encode()).hexdigest()] += 1
    top = max(sys_hashes.values())
    boiler_tokens = ntok(docs[0]["system"])

    marker_hits = 0
    malformed = 0
    for d in docs:
        a = d["asst"]
        for m in MARKERS:
            marker_hits += a.count(m)
        # pull thought + solution out from between the ghost markers
        th = re.search(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", a, re.S)
        so = re.search(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", a, re.S)
        thought = th.group(1).strip() if th else ""
        solution = so.group(1).strip() if so else ""
        if not (th and so):
            malformed += 1
            d["flags"].append("malformed_markers")
            # fall back to marker-stripped assistant text so we still keep signal
            stripped = a
            for m in MARKERS:
                stripped = stripped.replace(m, "")
            solution = solution or stripped.strip()
        # the clean training document: problem + reasoning + solution, no scaffold
        d["problem"] = d["user"].strip()
        d["text"] = f"{d['user'].strip()}\n\n{thought}\n\n{solution}".strip()
        d["tokens"] = ntok(d["text"])
        if "extract" not in EXAMPLES and th and so:
            EXAMPLES["extract"] = {
                "before": a[:200] + "  …",
                "after": (thought[:120] + " …").replace("\n", " "),
                "note": "system boilerplate (242 tok/row) + 4 ghost markers/row stripped; "
                        "thought/solution pulled out as clean training text"}
        d.pop("system"); d.pop("asst")

    stats["extract"] = {
        "rows_in": len(docs),
        "identical_system_boilerplate_rows": top,
        "boilerplate_tokens_per_row": boiler_tokens,
        "boilerplate_tokens_removed_total": boiler_tokens * len(docs),
        "ghost_markers_stripped": marker_hits,
        "malformed_trace_rows_flagged": malformed,
    }
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — NORMALIZE: NFC, entities, zero-width & control chars (Brahmic-safe)
# ─────────────────────────────────────────────────────────────────────────────
ZERO_WIDTH = {"​", "﻿", "­", "⁠", "᠎"}  # NOT ZWJ/ZWNJ
ZWJ, ZWNJ = "‍", "‌"
BRAHMIC = re2.compile(r"\p{Devanagari}|\p{Bengali}|\p{Tamil}|\p{Telugu}|"
                      r"\p{Kannada}|\p{Malayalam}|\p{Gujarati}|\p{Gurmukhi}|\p{Oriya}")
CTRL = re2.compile(r"[\p{Cc}\p{Cf}]")

def normalize_text(s):
    counters = defaultdict(int)
    before = s
    s = html.unescape(s)
    if s != before:
        counters["html_entities_unescaped"] += 1
    s = unicodedata.normalize("NFC", s)
    # remove zero-width chars (but keep ZWJ/ZWNJ only when adjacent to Brahmic)
    out = []
    for i, ch in enumerate(s):
        if ch in ZERO_WIDTH:
            counters["zero_width_removed"] += 1
            continue
        if ch in (ZWJ, ZWNJ):
            window = s[max(0, i-1):i+2]
            if BRAHMIC.search(window):
                out.append(ch)  # legitimate Brahmic joiner — keep
            else:
                counters["stray_joiner_removed"] += 1
            continue
        cat = unicodedata.category(ch)
        if (cat == "Cc" and ch not in "\n\t") or cat == "Cf":
            counters["control_char_removed"] += 1
            continue
        out.append(ch)
    s = "".join(out)
    # collapse runs of spaces/tabs and >2 blank lines
    s2 = re.sub(r"[ \t]+", " ", s)
    s2 = re.sub(r"\n{3,}", "\n\n", s2)
    s2 = "\n".join(ln.rstrip() for ln in s2.split("\n")).strip()
    if s2 != s:
        counters["whitespace_collapsed"] += 1
    return s2, counters

def stage_normalize(docs, stats):
    agg = defaultdict(int); docs_changed = 0; chars_removed = 0
    for d in docs:
        new, c = normalize_text(d["text"])
        if new != d["text"]:
            docs_changed += 1
            chars_removed += max(0, len(d["text"]) - len(new))
            if "normalize" not in EXAMPLES and (c.get("zero_width_removed") or c.get("control_char_removed") or c.get("html_entities_unescaped")):
                bad = [f"U+{ord(ch):04X}" for ch in d["text"] if ch in ZERO_WIDTH
                       or unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\n\t"]
                EXAMPLES["normalize"] = {
                    "codepoints_removed": sorted(set(bad))[:8],
                    "counts": dict(c),
                    "note": "invisible zero-width / control / BOM chars removed; HTML entities "
                            "unescaped; Brahmic ZWJ/ZWNJ preserved"}
            d["text"] = new
            d["problem"], _ = normalize_text(d["problem"])
            d["tokens"] = ntok(new)
        for k, v in c.items():
            agg[k] += v
    stats["normalize"] = {"docs_changed": docs_changed, "chars_removed": chars_removed, **agg}
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — LANGUAGE-ID: keep target langs; never trust the folder name
# ─────────────────────────────────────────────────────────────────────────────
EN_STOP = set("the of and to a in is that it for on with as are this be by an we you "
              "can not or if so we our will each into then they he she his her them".split())
LATIN = re2.compile(r"\p{Latin}")
def profile_lang(text):
    letters = re2.findall(r"\p{L}", text)
    if not letters:
        return "nonlingual", 1.0  # pure code/symbols — allowed (target: code)
    latin = sum(1 for ch in letters if LATIN.match(ch))
    latin_frac = latin / len(letters)
    words = re.findall(r"[a-zA-Z]+", text.lower())
    stop_hit = sum(1 for w in words if w in EN_STOP) / max(1, len(words))
    if latin_frac > 0.6 and (stop_hit > 0.03 or len(words) < 30):
        return "en", latin_frac
    if latin_frac <= 0.6:
        return "non-latin", latin_frac
    return "latin-other", latin_frac

def stage_langid(docs, stats):
    kept, dropped = [], defaultdict(int)
    dist = defaultdict(int)
    for d in docs:
        lang, frac = profile_lang(d["problem"] or d["text"][:2000])
        dist[lang] += 1
        if lang in ("en", "nonlingual"):
            d["lang"] = lang
            kept.append(d)
        else:
            dropped[lang] += 1
    stats["langid"] = {"kept": len(kept), "dropped": sum(dropped.values()),
                       "dropped_by_lang": dict(dropped), "language_dist": dict(dist)}
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — QUALITY: Gopher/C4 heuristics + reasoning-specific defects
# ─────────────────────────────────────────────────────────────────────────────
def quality_reasons(d):
    text = d["text"]; reasons = []
    words = text.split()
    wc = len(words)
    if wc < 20:
        reasons.append("too_short")
    # Gopher: symbol-to-word ratio
    sym = sum(text.count(c) for c in "#{}[]<>|")
    if wc and sym / wc > 0.5:
        reasons.append("symbol_heavy")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        bullets = sum(1 for ln in lines if ln.lstrip()[:1] in "-*•·")
        if bullets / len(lines) > 0.6:
            reasons.append("bullet_wall")
        ellip = sum(1 for ln in lines if ln.rstrip().endswith("..."))
        if ellip / len(lines) > 0.3:
            reasons.append("ellipsis_heavy")
        # duplicate-line fraction (degenerate / looping reasoning)
        uniq = len(set(lines))
        if 1 - uniq / len(lines) > 0.3:
            reasons.append("repeated_lines")
    # reasoning-specific: truncated / empty
    if "malformed_markers" in d["flags"] and wc < 40:
        reasons.append("truncated_trace")
    return reasons

def stage_quality(docs, stats):
    kept = []; by_reason = defaultdict(int)
    for d in docs:
        r = quality_reasons(d)
        if r:
            for x in r:
                by_reason[x] += 1
            continue
        kept.append(d)
    stats["quality"] = {"kept": len(kept), "dropped": len(docs) - len(kept),
                        "dropped_by_reason": dict(sorted(by_reason.items()))}
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — DEDUP: MinHash + LSH, implemented from the lecture's 3 steps
#   (1) shingle   (2) minhash signature   (3) LSH band + compare
# ─────────────────────────────────────────────────────────────────────────────
NUM_PERM, BANDS, ROWS_PER_BAND = 128, 32, 4  # 32*4 = 128
MPRIME = (1 << 31) - 1  # 2^31-1; keeps MH_A*hv (<2^62) inside uint64
def _mh_params(n, seed=0x5EED):
    import numpy as np
    x, rnd = seed, []
    for _ in range(n * 2):
        x = (1103515245 * x + 12345) & 0x7fffffff  # deterministic LCG
        rnd.append(x)
    a = np.array([1 + (rnd[i] % (MPRIME - 1)) for i in range(n)], dtype=np.uint64)
    b = np.array([rnd[n + i] % MPRIME for i in range(n)], dtype=np.uint64)
    return a, b
MH_A, MH_B = _mh_params(NUM_PERM)

def shingles(text, k=5):
    """Return a set of uint32 shingle hashes (overlapping word 5-grams)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        h = hashlib.md5(" ".join(words).encode()).digest()
        return {int.from_bytes(h[:4], "big")}
    return {int.from_bytes(hashlib.md5(" ".join(words[i:i+k]).encode()).digest()[:4], "big")
            for i in range(len(words) - k + 1)}

def minhash(sig_shingles):
    import numpy as np
    if not sig_shingles:
        return np.full(NUM_PERM, MPRIME, dtype=np.uint64)
    hv = (np.fromiter(sig_shingles, dtype=np.uint64) % MPRIME)  # (M,)
    # (num_perm, M) = a[:,None]*hv[None,:] + b[:,None]  (mod prime), then min over M
    vals = (MH_A[:, None] * hv[None, :] + MH_B[:, None]) % MPRIME
    return vals.min(axis=1)  # (num_perm,) uint64

class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[max(ra, rb)] = min(ra, rb)

def stage_dedup(docs, stats):
    sigs = [minhash(shingles(d["text"])) for d in docs]
    uf = UF(len(docs))
    for band in range(BANDS):
        buckets = defaultdict(list)
        for idx, sig in enumerate(sigs):
            key = (band, tuple(sig[band*ROWS_PER_BAND:(band+1)*ROWS_PER_BAND]))
            buckets[key].append(idx)
        for members in buckets.values():
            if len(members) > 1:
                for j in members[1:]:
                    uf.union(members[0], j)
    clusters = defaultdict(list)
    for i in range(len(docs)):
        clusters[uf.find(i)].append(i)
    keep_idx, removed, example = set(), 0, None
    for root, members in clusters.items():
        members.sort()
        keep_idx.add(members[0])
        if len(members) > 1:
            removed += len(members) - 1
            if example is None:
                example = {"kept_id": docs[members[0]]["id"],
                           "removed_id": docs[members[1]]["id"],
                           "kept_head": docs[members[0]]["problem"][:240],
                           "removed_head": docs[members[1]]["problem"][:240]}
    kept = [docs[i] for i in sorted(keep_idx)]
    dup_clusters = sum(1 for m in clusters.values() if len(m) > 1)
    stats["dedup"] = {"kept": len(kept), "near_dupes_removed": removed,
                      "duplicate_clusters": dup_clusters,
                      "params": {"num_perm": NUM_PERM, "bands": BANDS,
                                 "rows_per_band": ROWS_PER_BAND, "shingle_words": 5},
                      "example": example}
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — PII SCRUB: redact contact PII, keep public names (draw the line)
# ─────────────────────────────────────────────────────────────────────────────
# Precision-tuned for a code/math corpus: bare digit runs are DATA here, not PII.
# Phone requires real separators (so "1234567890" in a math problem is left alone);
# IPv4 validates octets 0-255 (so version strings like "1.2.3.999" don't match).
_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
PII = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("ipv4",  re.compile(rf"(?<![\d.]){_OCT}\.{_OCT}\.{_OCT}\.{_OCT}(?![\d.])")),
    ("phone", re.compile(r"(?<!\d)(?:\+\d{1,3}[ -])?(?:\(\d{3}\)[ -]?|\d{3}[ -])\d{3}[ -]\d{4}(?!\d)")),
    ("handle", re.compile(r"(?<![\w])@[A-Za-z0-9_]{3,}\b")),
]
def stage_pii(docs, stats):
    counts = defaultdict(int); docs_hit = 0; example = None
    for d in docs:
        hit = False
        for kind, rx in PII:
            def _sub(m):
                nonlocal hit
                counts[kind] += 1; hit = True
                return f"[{kind.upper()}]"
            new = rx.sub(_sub, d["text"])
            if new != d["text"] and example is None and kind == "email":
                m = rx.search(d["text"])
                if m:
                    s = max(0, m.start()-60)
                    example = {"kind": kind, "before": d["text"][s:m.end()+20],
                               "after": new[s:s+90]}
            d["text"] = new
        if hit:
            docs_hit += 1
            d["tokens"] = ntok(d["text"])
    stats["pii"] = {"docs_redacted": docs_hit, "redactions_by_type": dict(counts),
                    "policy": "redact contact PII (email/phone/ip/handle); keep public names",
                    "example": example}
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — DECONTAMINATE: n-gram overlap vs held-out benchmark test sets
# ─────────────────────────────────────────────────────────────────────────────
def norm_words(s):
    return re.findall(r"\w+", s.lower())
def ngrams(words, n=13):
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)} if len(words) >= n else set()
def informative(gram):
    """Reject low-entropy n-grams (digit/letter enumerations, single-char runs)
    that cause false-positive contamination hits — e.g. '0 1 2 3 4 5 ...'."""
    toks = gram.split()
    generic = sum(1 for t in toks if t.isdigit() or len(t) <= 1)
    distinct_words = {t for t in toks if len(t) > 1 and not t.isdigit()}
    return generic / len(toks) < 0.5 and len(distinct_words) >= 6

def stage_decontam(docs, stats):
    bench = json.loads((DATA / "benchmarks" / "benchmarks.json").read_text())
    bench_ng = {}  # ngram -> benchmark name
    for name, items in bench.items():
        for q in items:
            for g in ngrams(norm_words(q), 13):
                if informative(g):
                    bench_ng[g] = name
    kept, hits = [], defaultdict(int); example = None
    for d in docs:
        probe = ngrams(norm_words(d["problem"]), 13)
        overlap = [(g, bench_ng[g]) for g in probe if g in bench_ng]
        if len(overlap) >= 2:  # ≥2 shared 13-grams => contaminated
            src = overlap[0][1]
            hits[src] += 1
            if example is None:
                example = {"benchmark": src, "shared_13gram": overlap[0][0][:120],
                           "problem_head": d["problem"][:200]}
            continue
        kept.append(d)
    stats["decontam"] = {"kept": len(kept), "contaminated_removed": sum(hits.values()),
                         "by_benchmark": dict(hits),
                         "benchmarks": {k: len(v) for k, v in bench.items()},
                         "rule": "drop if ≥2 shared 13-grams with a benchmark test item",
                         "example": example}
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 — MANIFEST: per-shard provenance + reproducibility record
# ─────────────────────────────────────────────────────────────────────────────
def stage_manifest(docs, stats, t0):
    corpus = "\n".join(d["text"] for d in docs)
    corpus_sha = hashlib.sha256(corpus.encode()).hexdigest()
    final_tokens = sum(d["tokens"] for d in docs)
    langs = defaultdict(int)
    for d in docs:
        langs[d.get("lang", "en")] += 1
    manifest = {
        "shard_id": "openthoughts-s0-clean",
        "source": {"repo": SOURCE_REPO, "commit": SOURCE_COMMIT,
                   "license": SOURCE_LICENSE, "contributor": "open-thoughts",
                   "downloaded_via": "huggingface_hub hf_hub_download"},
        "script_sha256": SCRIPT_HASH,
        "input_sha256": sha256_file(SHARD),
        "output_sha256": corpus_sha,
        "rows": len(docs),
        "tokens": final_tokens,
        "tokenizer": "cl100k_base (tiktoken)",
        "languages": dict(langs),
        "pipeline": ["extract", "normalize", "langid", "quality", "dedup", "pii", "decontam"],
        "reproducible": "same input_sha256 + script_sha256 -> same output_sha256",
        "built_unix": int(t0),
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    stats["manifest"] = {"output_sha256": corpus_sha[:16], "final_rows": len(docs),
                         "final_tokens": final_tokens, "wrote": "report/manifest.json"}
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log(f"Loading {N_ROWS} rows from {SHARD.name} ...")
    docs = load_docs(N_ROWS)
    stats = {"config": {"source": SOURCE_REPO, "n_rows_loaded": len(docs),
                        "tokenizer": "cl100k_base"}}
    timeline = []
    def mark(stage, docs):
        toks = sum(d["tokens"] for d in docs)
        timeline.append({"stage": stage, "rows": len(docs), "tokens": toks})
        log(f"  [{stage:11}] rows={len(docs):6}  tokens={toks:,}")

    docs = stage_extract(docs, stats);      mark("extract", docs)
    docs = stage_normalize(docs, stats);    mark("normalize", docs)
    docs = stage_langid(docs, stats);       mark("langid", docs)
    docs = stage_quality(docs, stats);      mark("quality", docs)
    docs = stage_dedup(docs, stats);        mark("dedup", docs)
    docs = stage_pii(docs, stats);          mark("pii", docs)
    docs = stage_decontam(docs, stats);     mark("decontam", docs)
    manifest = stage_manifest(docs, stats, t0)

    raw0 = timeline[0]
    final = timeline[-1]
    stats["timeline"] = timeline
    stats["summary"] = {
        "rows_in": raw0["rows"], "rows_out": final["rows"],
        "rows_removed": raw0["rows"] - final["rows"],
        "rows_removed_pct": round(100*(raw0["rows"]-final["rows"])/raw0["rows"], 2),
        "tokens_in": raw0["tokens"], "tokens_out": final["tokens"],
        "tokens_removed_pct": round(100*(raw0["tokens"]-final["tokens"])/raw0["tokens"], 2),
        "elapsed_sec": round(time.time()-t0, 1),
    }
    (REPORT / "stats.json").write_text(json.dumps(stats, indent=2))
    # collect concrete before/after examples for the widget
    for st in ("dedup", "pii", "decontam"):
        if stats.get(st, {}).get("example"):
            EXAMPLES[st] = stats[st]["example"]
    (REPORT / "examples.json").write_text(json.dumps(EXAMPLES, indent=2))
    log(f"\nDONE in {stats['summary']['elapsed_sec']}s  "
        f"rows {raw0['rows']}->{final['rows']}  "
        f"tokens {raw0['tokens']:,}->{final['tokens']:,} "
        f"({stats['summary']['tokens_removed_pct']}% removed)")
    log("wrote report/stats.json + report/manifest.json")

if __name__ == "__main__":
    main()
