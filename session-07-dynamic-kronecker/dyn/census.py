"""The collision census: what a fixed byte window actually costs, per script.

This is the measurement Session 7 asked for — "take the V5 vocabulary, encode every
token, and count the collisions per script" (report §2h) — run against a real 131,072
BPE vocabulary trained on the Session 3 India-first corpus.

Two distinct damages are counted, because they are different failures:

  TRUNCATION — the token is longer than the window, so bytes are dropped. The token is
      still *represented*, just not faithfully.
  COLLISION  — two or more distinct tokens agree on their first `window` bytes, so they
      receive byte-identical codes and therefore identical embedding vectors. No amount
      of training can separate them; the projection is never shown a difference.

Collisions are the sovereign risk: truncation is lossy, collision is fatal.
"""
import collections
import json
import pathlib

import torch

from .codecs import ByteCodec
from .scripts import INDIC_SCRIPTS, bytes_per_char, token_script

ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOWS = (32, 48, 64)           # the three the session names


def load_vocab(path: pathlib.Path) -> list[str]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    vocab = tok.get_vocab()
    out = [""] * len(vocab)
    for s, i in vocab.items():
        out[i] = s
    return out


def census(vocab: list[str], windows=WINDOWS) -> dict:
    """Per-script truncation and collision counts at each candidate window."""
    scripts = [token_script(t) for t in vocab]
    raws = [t.encode("utf-8") for t in vocab]

    per_script = collections.Counter(scripts)
    byte_len = collections.defaultdict(list)
    for s, r in zip(scripts, raws):
        byte_len[s].append(len(r))

    result = {
        "vocab_size": len(vocab),
        "scripts": {},
        "windows": {},
    }
    for s, n in per_script.most_common():
        lens = byte_len[s]
        result["scripts"][s] = {
            "tokens": n,
            "bytes_per_char": bytes_per_char(s),
            "mean_byte_len": round(sum(lens) / len(lens), 2),
            "max_byte_len": max(lens),
        }

    for w in windows:
        groups = collections.defaultdict(list)
        for i, r in enumerate(raws):
            groups[r[:w]].append(i)

        trunc = collections.Counter()
        coll_tokens = collections.Counter()
        coll_clusters = collections.Counter()
        examples = []
        for key, members in groups.items():
            if len(members) > 1:
                # attribute the cluster to the dominant script of its members
                sc = collections.Counter(scripts[i] for i in members).most_common(1)[0][0]
                coll_clusters[sc] += 1
                coll_tokens[sc] += len(members)
                if len(examples) < 40 and sc in INDIC_SCRIPTS:
                    examples.append({
                        "script": sc,
                        "shared_prefix_bytes": w,
                        "tokens": [vocab[i] for i in members[:5]],
                    })
        for i, r in enumerate(raws):
            if len(r) > w:
                trunc[scripts[i]] += 1

        total_trunc = sum(trunc.values())
        total_coll = sum(coll_tokens.values())
        result["windows"][str(w)] = {
            "window_bytes": w,
            "truncated_tokens": total_trunc,
            "truncated_pct": round(100 * total_trunc / len(vocab), 3),
            "collision_clusters": sum(coll_clusters.values()),
            "collided_tokens": total_coll,
            "collided_pct": round(100 * total_coll / len(vocab), 3),
            "indic_share_of_collisions": round(
                100 * sum(v for k, v in coll_tokens.items() if k in INDIC_SCRIPTS)
                / max(1, total_coll), 1),
            "by_script": {
                s: {
                    "tokens": per_script[s],
                    "truncated": trunc.get(s, 0),
                    "truncated_pct": round(100 * trunc.get(s, 0) / per_script[s], 3),
                    "collided": coll_tokens.get(s, 0),
                    "collided_pct": round(100 * coll_tokens.get(s, 0) / per_script[s], 3),
                    "effective_chars": round(w / bytes_per_char(s), 1),
                }
                for s in per_script
                if trunc.get(s, 0) or coll_tokens.get(s, 0)
            },
            "examples": examples[:12],
        }
    return result


def colliding_clusters(vocab: list[str], window: int = 32) -> list[list[int]]:
    groups = collections.defaultdict(list)
    for i, t in enumerate(vocab):
        groups[t.encode("utf-8")[:window]].append(i)
    return [m for m in groups.values() if len(m) > 1]


def separation_report(vocab: list[str], clusters: list[list[int]],
                      codecs: dict[str, ByteCodec], max_clusters: int = 200) -> dict:
    """For tokens v1 *cannot* tell apart, how well does each codec separate them?

    Reports the worst-case (minimum) pairwise cosine distance within a cluster. A codec
    that collides scores 0.0 — meaning the two tokens are the same vector, forever.
    """
    out = {}
    sample = clusters[:max_clusters]
    for name, codec in codecs.items():
        worst = 1.0
        exact_collisions = 0
        margins = []
        for members in sample:
            toks = [vocab[i] for i in members]
            e = codec.encode(toks)
            e = torch.nn.functional.normalize(e, dim=-1)
            sim = e @ e.T
            n = len(members)
            off = sim[~torch.eye(n, dtype=torch.bool)]
            m = float((1.0 - off.max()).item())        # min cosine distance in cluster
            margins.append(m)
            worst = min(worst, m)
            if off.max() > 1 - 1e-6:
                exact_collisions += 1
        margins.sort()
        out[name] = {
            "codec": repr(codec),
            "code_dim": codec.code_dim,
            "clusters_tested": len(sample),
            "clusters_still_colliding": exact_collisions,
            "min_cosine_distance": round(worst, 6),
            "median_cosine_distance": round(margins[len(margins) // 2], 6) if margins else None,
        }
    return out


def morphological_pairs(vocab: list[str], n: int = 400, min_prefix: int = 4,
                        seed: int = 11) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build two control sets for the prefix-similarity check:

      related  — token pairs sharing a long prefix (`train`/`training`), which a byte
                 codec is *supposed* to place near each other;
      unrelated — random pairs sharing no prefix, as the baseline to beat.

    A codec that scores `related` no higher than `unrelated` has thrown away the
    spelling-awareness that motivates byte codecs in the first place.
    """
    import random
    rng = random.Random(seed)
    by_prefix = collections.defaultdict(list)
    for t in vocab:
        s = t.lstrip("▁")
        if len(s) >= min_prefix + 2 and s.isprintable():
            by_prefix[s[:min_prefix]].append(t)

    related = []
    for _, members in by_prefix.items():
        if len(members) >= 2:
            rng.shuffle(members)
            related.append((members[0], members[1]))
    rng.shuffle(related)
    related = related[:n]

    flat = [t for t in vocab if len(t) > 3]
    unrelated = []
    while len(unrelated) < len(related):
        a, b = rng.choice(flat), rng.choice(flat)
        if a.lstrip("▁")[:2] != b.lstrip("▁")[:2]:
            unrelated.append((a, b))
    return related, unrelated


def prefix_similarity(pairs_related, pairs_unrelated, codecs: dict[str, ByteCodec]) -> dict:
    out = {}
    for name, codec in codecs.items():
        row = {}
        for label, pairs in (("related", pairs_related), ("unrelated", pairs_unrelated)):
            a = codec.encode([p[0] for p in pairs])
            b = codec.encode([p[1] for p in pairs])
            a = torch.nn.functional.normalize(a, dim=-1)
            b = torch.nn.functional.normalize(b, dim=-1)
            row[label] = round(float((a * b).sum(-1).mean().item()), 4)
        row["separation"] = round(row["related"] - row["unrelated"], 4)
        out[name] = row
    return out


def main() -> int:
    tok_path = ROOT / "artifacts" / "tokenizer-131072.json"
    vocab = load_vocab(tok_path)
    print(f"vocab: {len(vocab):,} tokens")

    rep = census(vocab)
    w32 = rep["windows"]["32"]
    print(f"\nat window=32B: {w32['truncated_tokens']:,} truncated "
          f"({w32['truncated_pct']}%), {w32['collided_tokens']:,} collided "
          f"in {w32['collision_clusters']:,} clusters "
          f"({w32['indic_share_of_collisions']}% Indic)")

    clusters = colliding_clusters(vocab, 32)
    codecs = {
        "v1_onehot32": ByteCodec("onehot", 32),
        "v1_onehot64": ByteCodec("onehot", 64),
        "dyn_fourier16": ByteCodec("fourier", 16),
        "dyn_fourier32": ByteCodec("fourier", 32),
        "dyn_randproj16": ByteCodec("randproj", 16),
        "dyn_relative16": ByteCodec("relative", 16),
        "dyn_hybrid16": ByteCodec("hybrid", 16),
        "dyn_hybrid8": ByteCodec("hybrid", 8),
    }
    rep["separation"] = separation_report(vocab, clusters, codecs)
    for name, r in rep["separation"].items():
        print(f"  {name:16} dim={r['code_dim']:5} "
              f"still-colliding={r['clusters_still_colliding']:3}/{r['clusters_tested']}  "
              f"min_cos_dist={r['min_cosine_distance']}")

    rel, unrel = morphological_pairs(vocab)
    rep["prefix_similarity"] = prefix_similarity(rel, unrel, codecs)
    print(f"\nprefix similarity ({len(rel)} related vs {len(unrel)} unrelated pairs):")
    for name, r in rep["prefix_similarity"].items():
        print(f"  {name:16} related={r['related']:+.4f}  unrelated={r['unrelated']:+.4f}  "
              f"separation={r['separation']:+.4f}")

    out = ROOT / "artifacts" / "census.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
