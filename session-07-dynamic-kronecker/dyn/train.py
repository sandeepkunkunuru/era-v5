"""Experiment B: does the dynamic window cost anything on real language modelling?

The probe (probe.py) shows the dynamic codec fixes something v1 gets wrong. This shows
it does not break anything v1 gets right: identical transformer, identical untied output
head, identical data and seed — only the input path differs.

Data is real multilingual text from the Session 3 India-first corpus, tokenized with the
same 131,072 BPE vocabulary the census ran on, so the long Indic tokens that motivate the
whole exercise are actually present in the stream.
"""
import json
import pathlib
import time

import torch

from .census import load_vocab
from .codecs import ByteCodec
from .model import build

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent / "session-03-india-first-40b" / "data" / "train"
SEED = 707
# Indic-heavy on purpose: this is where the byte window bites.
LANGS = ["hi", "ta", "te", "ml", "bn", "mr", "kn", "gu", "or", "pa", "en"]
VOCAB_CAP = 12_000          # keeps the v1 code table (12k x 8192) on a laptop GPU


def load_stream(vocab_all: list[str], chars_per_lang: int = 260_000):
    """Tokenize a slice of each language, then remap to the most frequent ids."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "artifacts" / "tokenizer-131072.json"))

    ids: list[int] = []
    for lang in LANGS:
        p = CORPUS / f"wiki_{lang}.txt"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")[:chars_per_lang]
        ids.extend(tok.encode(text).ids)

    counts = torch.bincount(torch.tensor(ids), minlength=len(vocab_all))
    keep = torch.topk(counts, VOCAB_CAP).indices.tolist()
    remap = {old: new for new, old in enumerate(keep)}
    unk = len(keep)
    sub_vocab = [vocab_all[i] for i in keep] + ["<UNK>"]
    stream = torch.tensor([remap.get(i, unk) for i in ids], dtype=torch.long)

    covered = float((stream != unk).float().mean().item())
    return sub_vocab, stream, covered


def batches(stream: torch.Tensor, block: int, batch: int, device: str, gen):
    n = len(stream) - block - 1
    while True:
        ix = torch.randint(0, n, (batch,), generator=gen)
        x = torch.stack([stream[i:i + block] for i in ix]).to(device)
        y = torch.stack([stream[i + 1:i + 1 + block] for i in ix]).to(device)
        yield x, y


def run_arm(name: str, codec: ByteCodec | None, sub_vocab, train_s, val_s,
            device: str, steps: int, d_model: int, block: int, batch: int) -> dict:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = build("dense" if codec is None else "kron", sub_vocab, d_model,
                  codec=codec, n_layer=4, n_head=4, block_size=block).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps,
                                                pct_start=0.1)

    it = batches(train_s, block, batch, device, gen)
    t0 = time.time()
    for step in range(steps):
        x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    vgen = torch.Generator().manual_seed(SEED + 1)
    vit = batches(val_s, block, batch, device, vgen)
    with torch.no_grad():
        losses = [float(model(*next(vit))[1].item()) for _ in range(30)]
    val = sum(losses) / len(losses)
    model.train()

    return {
        "arm": name,
        "codec": repr(codec) if codec else "dense nn.Embedding",
        "input_path_params": model.input_path_params,
        "total_trainable": model.trainable_params(),
        "val_loss": round(val, 4),
        "val_ppl": round(float(torch.exp(torch.tensor(val)).item()), 2),
        "seconds": round(time.time() - t0, 1),
    }


def main(steps: int = 1200, d_model: int = 256, block: int = 128, batch: int = 16) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_all = load_vocab(ROOT / "artifacts" / "tokenizer-131072.json")
    sub_vocab, stream, covered = load_stream(vocab_all)
    split = int(0.9 * len(stream))
    train_s, val_s = stream[:split], stream[split:]
    print(f"stream: {len(stream):,} tokens, vocab {len(sub_vocab):,}, "
          f"coverage {covered:.1%}, device {device}")

    arms = {
        "dense_control": None,
        "kron_v1_onehot32": ByteCodec("onehot", 32),
        "dyn_fourier16": ByteCodec("fourier", 16),
        "dyn_hybrid8": ByteCodec("hybrid", 8),
    }
    results = {"steps": steps, "d_model": d_model, "block": block, "batch": batch,
               "vocab": len(sub_vocab), "stream_tokens": len(stream),
               "device": device, "arms": []}
    for name, codec in arms.items():
        r = run_arm(name, codec, sub_vocab, train_s, val_s, device,
                    steps, d_model, block, batch)
        results["arms"].append(r)
        print(f"  {name:18} val_loss {r['val_loss']:.4f}  ppl {r['val_ppl']:8.2f}  "
              f"input-path {r['input_path_params']:>9,}  ({r['seconds']}s)")

    base = next(a for a in results["arms"] if a["arm"] == "kron_v1_onehot32")
    for a in results["arms"]:
        a["input_path_vs_v1"] = round(a["input_path_params"] / base["input_path_params"], 4)
        a["val_loss_delta_vs_v1"] = round(a["val_loss"] - base["val_loss"], 4)

    out = ROOT / "artifacts" / "lm.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
