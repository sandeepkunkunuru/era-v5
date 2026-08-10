"""The decisive experiment: a task Kronecker v1 provably cannot learn.

Take real token pairs from the V5 vocabulary that collide at a 32-byte window — for
example the Tamil inflections ▁பயன்படுத்திய and ▁பயன்படுத்தும் ("used" vs "that uses"),
which agree on their first 32 bytes and differ only after. Build the smallest possible
task that requires telling them apart:

    input  [X]           X is either the A-member or the B-member of a colliding pair
    target  ans_A if X is the A-member, else ans_B

Under v1 the two inputs are the *same vector*. Not similar — identical, bit for bit.
The network is therefore a function of an input that carries no information about the
label, so its accuracy is pinned at exactly 50% no matter how long it trains. This is
the same experimental shape Session 2 used to demonstrate that a token-only model
cannot learn order (report §2j), and it is falsifiable in the same way: if v1 ever
exceeded chance, the argument would be wrong.

The script asserts the identity numerically before training, so the claim is verified
rather than assumed.
"""
import json
import pathlib

import torch

from .census import colliding_clusters, load_vocab
from .codecs import ByteCodec
from .model import build

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 707


def make_task(vocab_all: list[str], n_pairs: int = 24, window: int = 32):
    """Build a tiny vocabulary of real colliding pairs plus two answer tokens."""
    clusters = [c for c in colliding_clusters(vocab_all, window) if len(c) >= 2]
    pairs = [(vocab_all[c[0]], vocab_all[c[1]]) for c in clusters[:n_pairs]]

    toks: list[str] = []
    for a, b in pairs:
        toks += [a, b]
    answers = ["<ANS_A>", "<ANS_B>"]
    vocab = toks + answers
    idx = {t: i for i, t in enumerate(vocab)}

    xs, ys = [], []
    for a, b in pairs:
        xs.append(idx[a]); ys.append(idx["<ANS_A>"])
        xs.append(idx[b]); ys.append(idx["<ANS_B>"])
    return vocab, pairs, torch.tensor(xs), torch.tensor(ys)


def verify_collision(vocab: list[str], pairs, codec: ByteCodec) -> dict:
    """Numerically confirm whether a codec maps each pair to the same vector."""
    a = codec.encode([p[0] for p in pairs])
    b = codec.encode([p[1] for p in pairs])
    delta = (a - b).abs().max(dim=-1).values
    return {
        "max_abs_difference": round(float(delta.max().item()), 10),
        "pairs_bit_identical": int((delta == 0).sum().item()),
        "pairs_total": len(pairs),
    }


def run_arm(name: str, vocab: list[str], xs: torch.Tensor, ys: torch.Tensor,
            codec: ByteCodec | None, device: str, steps: int = 400) -> dict:
    torch.manual_seed(SEED)
    kind = "dense" if codec is None else "kron"
    model = build(kind, vocab, d_model=64, codec=codec,
                  n_layer=2, n_head=2, block_size=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    X = xs.unsqueeze(1).to(device)                 # [N,1] one-token sequences
    Y = ys.unsqueeze(1).to(device)
    best = 0.0
    for _ in range(steps):
        logits, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc = float((logits.argmax(-1) == Y).float().mean().item())
        best = max(best, acc)

    with torch.no_grad():
        logits, loss = model(X, Y)
        final = float((logits.argmax(-1) == Y).float().mean().item())
    return {
        "arm": name,
        "input_path_params": model.input_path_params,
        "final_accuracy": round(final, 4),
        "best_accuracy": round(best, 4),
        "final_loss": round(float(loss.item()), 4),
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_all = load_vocab(ROOT / "artifacts" / "tokenizer-131072.json")
    vocab, pairs, xs, ys = make_task(vocab_all)
    print(f"probe: {len(pairs)} real colliding pairs from the 131,072 vocabulary")
    for a, b in pairs[:4]:
        print(f"   {a!r}  vs  {b!r}")

    arms = {
        "dense_control": None,
        "kron_v1_onehot32": ByteCodec("onehot", 32),
        "dyn_hybrid8": ByteCodec("hybrid", 8),
        "dyn_fourier16": ByteCodec("fourier", 16),
    }

    results = {"n_pairs": len(pairs), "device": device,
               "examples": [[a, b] for a, b in pairs[:8]], "arms": []}
    print()
    for name, codec in arms.items():
        row = {"name": name}
        if codec is not None:
            row["collision_check"] = verify_collision(vocab, pairs, codec)
            ident = row["collision_check"]["pairs_bit_identical"]
            print(f"{name:18} bit-identical pairs: {ident}/{len(pairs)}", end="  ")
        else:
            print(f"{name:18} (trainable row per token)      ", end="  ")
        row.update(run_arm(name, vocab, xs, ys, codec, device))
        print(f"-> accuracy {row['final_accuracy']:.3f}  "
              f"(input-path params {row['input_path_params']:,})")
        results["arms"].append(row)

    out = ROOT / "artifacts" / "probe.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
