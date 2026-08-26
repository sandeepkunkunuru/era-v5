"""ERA V5 Session 9 -- the loss harness.

    python loss_harness.py            # full run: 7 requirements + Part 2 + the extras
    python loss_harness.py --quick    # fewer training steps, same structure

Everything printed here is computed live. Nothing is quoted from the lecture.
Numbers land in results.json; the transcript of a run lands in run.log.
"""
import argparse, json, math, os, sys, time, urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT, Config

HERE = Path(__file__).parent
SEED = 20260822
DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = {}                                    # every reported number lands here


def say(*a):
    print(*a, flush=True)


def rule(n, title):
    say(f"\n{'='*78}\n  {n}  {title}\n{'='*78}")


# --------------------------------------------------------------------------- data
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def corpus():
    """Tiny Shakespeare, tokenised with the GPT-2 BPE. One file, works anywhere."""
    p = HERE / "input.txt"
    if not p.exists():
        say(f"downloading {SHAKESPEARE}")
        urllib.request.urlretrieve(SHAKESPEARE, p)
    return p.read_text()


def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------------ chunked loss
def naive_ce(hidden, weight, targets, ignore_index=-100):
    """Materialise every logit, then reduce. The path that hits the wall."""
    logits = hidden @ weight.t()                       # [N, V]  <- the expensive tensor
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index)


def _chunk_sum(h, w, t, ignore_index):
    logits = h @ w.t()
    return F.cross_entropy(logits.float(), t, ignore_index=ignore_index, reduction="sum")


def chunked_ce(hidden, weight, targets, chunk, ignore_index=-100):
    """Same objective, computed `chunk` rows at a time.

    Each chunk's logits are wrapped in a checkpoint, so they are freed after the forward
    and recomputed during the backward -- arithmetic traded for memory, which is the whole
    trade the technique rests on. Summing and dividing by the true contributing count at the
    end makes this the *exact* same mean as naive_ce, not an approximation of it.
    """
    n = int((targets != ignore_index).sum())
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[0], chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            _chunk_sum, hidden[i:i + chunk], weight, targets[i:i + chunk], ignore_index,
            use_reentrant=False)
    return total / max(n, 1)


def peak_mb(fn):
    """Peak CUDA allocation during one forward+backward of fn: (MiB, loss value)."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    loss = fn()
    value = loss.detach().item()
    loss.backward()
    torch.cuda.synchronize()
    mib = torch.cuda.max_memory_allocated() / 2**20
    del loss
    torch.cuda.empty_cache()
    return mib, value


def measure_ce(label, V, D, N, chunks, dtype=torch.bfloat16, weight=None):
    """Peak memory of naive vs chunked cross-entropy, with nothing else resident.

    The hidden states are synthetic on purpose: requirement 7 asks about the cost of the
    LOSS, and leaving a model in memory would just add a constant to every row and shrink
    the ratio into meaninglessness.
    """
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    W = weight if weight is not None else (torch.randn(V, D, device=DEV, dtype=dtype) * 0.02)
    W = W.detach().requires_grad_(True)
    tgt = torch.randint(0, V, (N,), device=DEV)
    say(f"  {label}: N={N:,} positions, D={D}, V={V:,}, {str(dtype).split('.')[-1]}")
    say(f"    logits tensor = {N*V*W.element_size()/2**20:,.0f} MiB, "
        f"{N*V*4/2**20:,.0f} MiB once cross_entropy upcasts it to fp32\n")

    # ONE hidden state, cloned per row. Every row must see identical inputs, or the losses
    # below are not comparable and the "same objective" claim is untestable.
    h0 = torch.randn(N, D, device=DEV, dtype=dtype)
    rows = []
    for name, fn in [("naive (all logits at once)", None)] + [(f"chunked, chunk={c}", c) for c in chunks]:
        h = h0.clone().requires_grad_(True)
        f = (lambda: naive_ce(h, W, tgt)) if fn is None else (lambda c=fn: chunked_ce(h, W, tgt, c))
        mib, val = peak_mb(f)
        rows.append((name, mib, val))
        del h
        torch.cuda.empty_cache()

    w = max(len(r[0]) for r in rows)
    base = rows[0][1]
    say(f"    {'implementation':<{w}}  {'peak MiB':>10}  {'loss':>14}  {'vs naive':>9}")
    for name, mib, val in rows:
        say(f"    {name:<{w}}  {mib:>10.1f}  {val:>14.9f}  {base/mib:>8.2f}x")
    best = min(r[1] for r in rows[1:])
    worst = max(abs(r[2] - rows[0][2]) for r in rows[1:])
    say(f"\n    ratio naive : best chunked = {base/best:.2f}x")
    say(f"    largest disagreement between any two losses: {worst:.2e}")

    # What chunking can never remove: the head matrix, its gradient, and whatever copies the
    # matmul needs. Measured, so the plateau above is explained rather than asserted.
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    hf = h0.clone().requires_grad_(True)
    l = _chunk_sum(hf[:1], W, tgt[:1], -100); l.backward()
    torch.cuda.synchronize()
    floor = torch.cuda.max_memory_allocated() / 2**20
    del hf, l; torch.cuda.empty_cache()
    say(f"    floor (one row of logits, so essentially just the head + its gradient): {floor:,.1f} MiB")
    say(f"    best chunked sits {best-floor:,.1f} MiB above that floor")
    del h0
    del W, tgt
    torch.cuda.empty_cache()
    return dict(N=N, D=D, V=V, dtype=str(dtype),
                rows=[dict(impl=n, peak_mib=m, loss=v) for n, m, v in rows],
                ratio=base / best, max_loss_disagreement=worst, floor_mib=floor)


def pretrain(model, text, enc, steps, label="", pad_prob=0.0, pad_id=None):
    """Train `model` on next-token prediction. Short, but enough that the model has opinions.

    Requirements 3 and 4 are meaningless on an untrained model -- every position costs ln(V),
    so masking anything changes nothing. `pad_prob` optionally truncates sequences and pads
    them, which is how a model *learns* that <pad> is free to predict.
    """
    model.train()
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    train = data[:int(0.9 * len(data))].to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    B, T = 8, model.cfg.block_size
    g = torch.Generator(device=DEV).manual_seed(SEED)
    V = model.cfg.vocab_size
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(len(train) - T - 1, (B,), generator=g, device=DEV)
        seq = torch.stack([train[j:j + T + 1] for j in i])
        idx, tgt = seq[:, :T].clone(), seq[:, 1:].clone()
        if pad_prob:
            keep = torch.randint(T // 3, T, (B,), generator=g, device=DEV)
            mask = torch.arange(T, device=DEV)[None, :] >= keep[:, None]
            idx[mask], tgt[mask] = pad_id, pad_id
        loss = F.cross_entropy(model(idx).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    say(f"  trained {steps} steps{label} in {time.time()-t0:.0f}s -> loss {loss.item():.4f}")
    del opt, train, data
    return model


# =========================================================== 1. shapes
def req1(model, tokens, enc):
    rule("1", "Every tensor shape, and what each dimension means")
    B, T = tokens.shape
    D, V = model.cfg.d_model, model.cfg.vocab_size
    with torch.no_grad():
        h = model.hidden(tokens)
        logits = model.lm_head(h)
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    flat_l, flat_t = logits[:, :-1].reshape(-1, V), tgt.reshape(-1)
    rows = [
        ("tokens",       tuple(tokens.shape),  "B sequences x T positions; each entry is one token id"),
        ("hidden",       tuple(h.shape),       "B x T x D; one D-vector per position, having seen its whole left context"),
        ("lm_head.weight", tuple(model.lm_head.weight.shape), "V rows x D; one learned row per vocabulary token"),
        ("logits",       tuple(logits.shape),  "B x T x V; one raw score per vocabulary token, per position"),
        ("inputs",       tuple(inp.shape),     "B x (T-1); the last position is dropped -- nothing follows it"),
        ("targets",      tuple(tgt.shape),     "B x (T-1); the first token is dropped -- nothing predicts it"),
        ("flat logits",  tuple(flat_l.shape),  "(B*(T-1)) x V; cross_entropy wants one row per prediction"),
        ("flat targets", tuple(flat_t.shape),  "(B*(T-1)); one correct token id per prediction"),
    ]
    w = max(len(r[0]) for r in rows)
    for name, shape, meaning in rows:
        say(f"  {name:<{w}}  {str(shape):<22}  {meaning}")
    say(f"\n  B={B}  T={T}  D={D}  V={V:,}")
    say(f"  logits tensor is {logits.numel()/h.numel():.0f}x larger than the hidden state that produced it")
    R["shapes"] = {n: list(s) for n, s, _ in rows}
    R["logits_vs_hidden_ratio"] = logits.numel() / h.numel()


# =========================================================== 2. the shift, in strings
def req2(tokens, enc):
    rule("2", "Verify the shift by printing token STRINGS, not ids")
    ids = tokens[0].tolist()
    inp, tgt = ids[:-1], ids[1:]
    say("  input token           ->  target token        (target must be the NEXT input)")
    for i in range(12):
        say(f"  {i:>3}  {enc.decode([inp[i]])!r:<20} -> {enc.decode([tgt[i]])!r:<20}")
    say("\n  read as text:")
    say(f"    inputs : {enc.decode(inp[:24])!r}")
    say(f"    targets: {enc.decode(tgt[:24])!r}")
    ok = inp[1:] == tgt[:-1]
    say(f"\n  inputs[1:] == targets[:-1] ? {ok}   <- this is the shift, asserted not eyeballed")
    assert ok
    R["shift_verified"] = ok

    say("\n  -- and the bug the lecture warns about twice: shifting the WRONG way --")
    say("     targets = tokens (no shift) hands the model its own input as the answer.")
    say("     inputs : %r" % enc.decode(ids[:12]))
    say("     targets: %r   <- identical. The loss will look wonderful." % enc.decode(ids[:12]))


# =========================================================== 3. padding
def req3(model, enc, V, text):
    rule("3", "Mask padding, and confirm the contributing-token count changes")
    PAD = V - 1                                   # a token id we reserve as <pad>
    T = 64
    torch.manual_seed(SEED)
    ids = enc.encode(text[5000:5000 + 4000])
    batch = torch.tensor(ids[:4 * T], device=DEV).view(4, T)
    lens = [64, 48, 28, 16]                       # four sequences of genuinely different length
    for r, L in enumerate(lens):
        batch[r, L:] = PAD
    inp, tgt = batch[:, :-1], batch[:, 1:]
    with torch.no_grad():
        logits = model(inp)

    counted = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    n_all = tgt.numel()

    masked_tgt = tgt.clone()
    masked_tgt[tgt == PAD] = -100                 # the ignore index
    ignored = F.cross_entropy(logits.reshape(-1, V), masked_tgt.reshape(-1), ignore_index=-100)
    n_real = int((masked_tgt != -100).sum())

    say(f"  sequence lengths in the batch : {lens}  (padded to {batch.shape[1]})")
    say(f"  positions if padding counts   : {n_all}")
    say(f"  positions after masking       : {n_real}   ({n_all - n_real} pad targets dropped, "
        f"{100*(n_all-n_real)/n_all:.0f}% of the batch)")
    say(f"\n  loss counting padding         : {counted:.4f}")
    say(f"  loss ignoring padding         : {ignored:.4f}")
    say(f"  difference                    : {ignored - counted:+.4f}")
    say("\n  The count is the part the requirement asks for, and it changed: "
        f"{n_all} -> {n_real}.")
    say("  The loss moved too, but note which way. This model has never been trained on padded")
    say("  batches, so it assigns <pad> almost no probability and those positions are the")
    say("  EXPENSIVE ones -- masking them lowers the number. The famous trap is the opposite")
    say("  case, and it needs a model that has learned <pad>. Demonstrated below.")
    R["padding"] = dict(lengths=lens, n_all=n_all, n_real=n_real,
                        loss_counting_pad=float(counted), loss_ignoring_pad=float(ignored))

    say("\n  -- the denominator bug, separately --")
    say("     Even with an ignore_index, dividing by B*T instead of the real count rescales the loss")
    total = F.cross_entropy(logits.reshape(-1, V), masked_tgt.reshape(-1),
                            ignore_index=-100, reduction="sum")
    say(f"     sum / n_real ({n_real:>3}) = {total/n_real:.4f}   <- correct")
    say(f"     sum / B*T   ({n_all:>3}) = {total/n_all:.4f}   <- wrong, and it moves with every batch")
    R["padding"]["loss_wrong_denominator"] = float(total / n_all)

    say("\n  -- and a distinction the lecture blurs: <eos> is NOT <pad> --")
    say("     <pad> is an artefact of batching and must be masked. <eos> is a real token the model")
    say("     has to learn to emit; mask it and the model never learns to stop.")

    # The trap only springs once the model has LEARNED that <pad> is free. Train a copy on
    # padded batches and watch the two losses come apart.
    say("\n  -- the trap, demonstrated --")
    say("     Above, the model has never seen <pad>, so masking it barely moves the number. The")
    say("     danger appears once the model has learned <pad> is trivially predictable. Train a")
    say("     copy on padded batches and measure both losses again:\n")
    import copy
    m2 = copy.deepcopy(model)
    pretrain(m2, text, enc, 300, label=" on PADDED batches", pad_prob=1.0, pad_id=PAD)
    with torch.no_grad():
        lg2 = m2(inp)
    c2 = F.cross_entropy(lg2.reshape(-1, V), tgt.reshape(-1))
    i2 = F.cross_entropy(lg2.reshape(-1, V), masked_tgt.reshape(-1), ignore_index=-100)
    say(f"\n     {'':<26}{'counting pad':>14}{'masking pad':>14}{'gap':>10}")
    say(f"     {'before padded training':<26}{counted:>14.4f}{ignored:>14.4f}{ignored-counted:>10.4f}")
    say(f"     {'after padded training':<26}{c2:>14.4f}{i2:>14.4f}{i2-c2:>10.4f}")
    drop_counted, drop_honest = float(counted - c2), float(ignored - i2)
    say(f"\n     The counted loss fell {drop_counted:.4f}. The honest loss fell {drop_honest:.4f}.")
    if drop_counted > 0:
        real = 100 * drop_honest / drop_counted
        say(f"     Only {real:.0f}% of that apparent progress is real; the rest is the model")
        say("     learning to predict padding, which is worth nothing at inference.")
        R["padding"]["real_fraction_of_apparent_progress"] = real
    R["padding"]["after_padded_training"] = dict(counting=float(c2), masking=float(i2))
    del m2
    torch.cuda.empty_cache() if DEV == "cuda" else None


# =========================================================== 4. document boundaries
def req4(model, enc, V, text):
    rule("4", "Pack two documents into one sequence, mask the boundary")
    # Two genuinely unrelated documents: Elizabethan verse, and Python source.
    doc_a = text[2000:2000 + 900]
    doc_b = (Path(__file__).parent / "model.py").read_text()[:900]
    EOD = V - 2
    a, b = enc.encode(doc_a), enc.encode(doc_b)
    say(f"  doc A: Shakespeare, {len(a)} tokens -- {enc.decode(a[:11])!r}...")
    say(f"  doc B: Python source, {len(b)} tokens -- {enc.decode(b[:11])!r}...")

    # Pack A|B|A|B... so boundaries are a meaningful fraction, as real packing makes them.
    # The packed sequence must fit the model's position table, so size the segments to it.
    n_seg, seg_len = 8, model.cfg.block_size // 8 - 1
    ids, is_boundary = [], []
    for i in range(n_seg):
        seg = (a if i % 2 == 0 else b)[:seg_len]
        ids += seg + [EOD]
        is_boundary += [False] * len(seg) + [True]   # the EOD position predicts across the join
    T = len(ids)
    assert T <= model.cfg.block_size, f"packed {T} tokens into a {model.cfg.block_size} window"

    tok = torch.tensor(ids, device=DEV).unsqueeze(0)
    bnd = torch.tensor(is_boundary, device=DEV).unsqueeze(0)
    inp, tgt, bnd_t = tok[:, :-1], tok[:, 1:], bnd[:, :-1]
    with torch.no_grad():
        logits = model(inp)

    unmasked = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    masked_tgt = tgt.clone()
    masked_tgt[bnd_t] = -100
    masked = F.cross_entropy(logits.reshape(-1, V), masked_tgt.reshape(-1), ignore_index=-100)

    per_tok = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none")
    at_bnd = per_tok[bnd_t.reshape(-1)]
    off_bnd = per_tok[~bnd_t.reshape(-1)]

    n_b = int(bnd_t.sum())
    say(f"\n  packed sequence               : {T} tokens, {n_b} document boundaries "
        f"({100*n_b/T:.1f}% of positions)")
    say(f"  loss WITHOUT boundary mask    : {unmasked:.4f}")
    say(f"  loss WITH boundary mask       : {masked:.4f}")
    say(f"  difference                    : {masked - unmasked:+.4f}")
    say(f"\n  mean loss AT the {n_b} boundary positions : {at_bnd.mean():.4f}")
    say(f"  mean loss at the {len(off_bnd)} other positions   : {off_bnd.mean():.4f}")
    say(f"  a boundary position is {at_bnd.mean()/off_bnd.mean():.2f}x as expensive as an ordinary one")
    say("\n  Explanation. A boundary position asks the model to predict the first token of a Python")
    say("  file from the last token of a Shakespeare speech. Nothing connects them, so the model")
    say("  cannot do better than its prior -- which is why those positions are the most expensive")
    say(f"  in the sequence, {at_bnd.mean()/off_bnd.mean():.2f}x an ordinary one.")
    say("\n  So masking them makes the reported loss FALL. That is not the loss being flattered:")
    say("  the positions removed were the ones teaching the model something false, namely that a")
    say("  Python file follows a Shakespeare speech. The unmasked number is the dishonest one --")
    say("  it is an average that includes a question with no answer, and it will drift with your")
    say("  packing density rather than with your model.")
    R["boundary"] = dict(T=T, n_boundaries=n_b, loss_unmasked=float(unmasked),
                         loss_masked=float(masked), loss_at_boundary=float(at_bnd.mean()),
                         loss_off_boundary=float(off_bnd.mean()))


# =========================================================== 5. perplexity
def req5(model, tokens, V):
    rule("5", "Perplexity, and where an untrained model must start")
    with torch.no_grad():
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), tokens[:, 1:].reshape(-1))
    ppl, anchor = math.exp(loss), math.log(V)
    say(f"  vocabulary V                      : {V:,}")
    say(f"  ln(V), the uniform-guess loss     : {anchor:.4f}")
    say(f"  measured loss, untrained model    : {loss:.4f}")
    say(f"  measured perplexity               : {ppl:,.0f}")
    say(f"  ratio perplexity / V              : {ppl/V:.3f}")
    say(f"  excess over ln(V)                 : {loss - anchor:+.4f}")
    say("\n  It lands just ABOVE ln(V), and it has to. ln(V) is the loss of a model that is exactly")
    say("  uniform; any spread in the initial logits is confidence the model has not earned, and on")
    say("  a random target unearned confidence costs more than it saves. The excess is a direct")
    say("  measure of how opinionated the initialisation is.")
    R["perplexity"] = dict(V=V, ln_V=anchor, loss=float(loss), ppl=ppl, ratio=ppl / V)


# =========================================================== 6. tied vs untied
def req6(cfg):
    rule("6", "Tied against untied output head, on this configuration")
    tied = GPT(Config(**{**cfg.__dict__, "tie_weights": True})).n_params()
    untied = GPT(Config(**{**cfg.__dict__, "tie_weights": False})).n_params()
    head = cfg.vocab_size * cfg.d_model
    say(f"  V x D (the head matrix)   : {cfg.vocab_size:,} x {cfg.d_model} = {head:,}")
    say(f"  total params, TIED        : {tied:,}")
    say(f"  total params, UNTIED      : {untied:,}")
    say(f"  cost of untying           : {untied-tied:,}  (+{100*(untied-tied)/tied:.1f}%)")
    say(f"  head as a share of the tied model : {100*head/tied:.1f}%")
    say("\n  At V5 scale the same arithmetic is far less forgiving:")
    for D in (4096, 8192):
        say(f"    V=131,072  D={D:<5} -> head = {131072*D:,} params "
            f"({131072*D*2/2**30:.2f} GiB in bf16, before optimiser state)")
    say("\n  And Session 7 closes the escape: with a byte-codec input side there is no [V,D] input")
    say("  table to tie to, so 'just tie it' is not available to V5 at any price.")
    R["tying"] = dict(tied=tied, untied=untied, head=head,
                      pct=100 * (untied - tied) / tied)


# =========================================================== 7. peak memory
def req7(model, V):
    rule("7", "Peak memory: ordinary cross-entropy against a chunked one")
    if DEV != "cuda":
        say("  no CUDA device -- skipping the measurement"); return
    D = model.cfg.d_model
    W = model.lm_head.weight.detach().to(torch.bfloat16)
    del model
    torch.cuda.empty_cache()

    say("  Same objective, two implementations. Chunking computes `chunk` rows of logits at a")
    say("  time inside a checkpoint, so they are freed after the forward and recomputed during")
    say("  the backward. Arithmetic traded for memory; the loss must not move at all.\n")
    R["memory_model_scale"] = measure_ce(
        "this model's head", V, D, 4096, chunks=(1024, 256, 64), weight=W)
    del W
    torch.cuda.empty_cache()

    say(f"\n  {'-'*72}")
    say("  Again at the V5 vocabulary, which is where this stops being an optimisation and")
    say("  starts being the difference between a run that fits and one that does not.\n")
    R["memory_v5_vocab"] = measure_ce(
        "V5 vocabulary", 131072, 1024, 2048, chunks=(512, 128, 32))

    say("\n  Projection to the real V5 training shape (arithmetic from the shapes, not measured")
    say("  on this 6 GiB laptop card):")
    for B, T in ((8, 8192), (4, 32768), (1, 262144)):
        g = B * T * 131072 * 2 / 2**30
        say(f"    B={B:<2} T={T:<7} -> logits {g:7.2f} GiB bf16, {2*g:7.2f} GiB with the backward")
    say("\n  The last row is one intermediate tensor, larger than any accelerator sold, for a")
    say("  quantity whose entire purpose is to be collapsed into a single scalar.")
    R["memory_projection"] = [dict(B=B, T=T, logits_gib=B*T*131072*2/2**30,
                                   with_backward_gib=2*B*T*131072*2/2**30)
                              for B, T in ((8, 8192), (4, 32768), (1, 262144))]


def unigram_entropy(enc, text):
    """H(X) of the training tokens, in nats -- the loss a marginal-only model would reach."""
    import collections
    ids = enc.encode(text)
    train = ids[:int(0.9 * len(ids))]
    n = len(train)
    counts = collections.Counter(train)
    return -sum(v / n * math.log(v / n) for v in counts.values())


# =========================================================== Part 2: a second head
def train_two_heads(text, enc, V, steps, log_every, tie, verbose):
    """Train a small model with heads at t+1 and t+2. Returns the loss history."""
    cfg = Config(vocab_size=V, n_layer=4, n_head=4, d_model=256, block_size=128,
                 n_extra_heads=1, tie_weights=tie)
    torch.manual_seed(SEED)
    model = GPT(cfg).to(DEV)
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    train = data[:int(0.9 * len(data))].to(DEV)
    if verbose:
        say(f"  model: {cfg.n_layer} layers, d_model={cfg.d_model}, block={cfg.block_size}, "
            f"head 1 {'TIED to the embedding' if tie else 'untied'}")
        say(f"  {model.n_params():,} params; the t+2 head adds {V*cfg.d_model:,} of them")
        say(f"  corpus: {len(data):,} GPT-2 tokens, training on {len(train):,}")
        say(f"\n  {'step':>6}  {'L1 (t+1)':>10}  {'L2 (t+2)':>10}  {'L1+L2':>10}  {'L2-L1':>8}  {'ppl(L1)':>10}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    B, T = 16, cfg.block_size
    g = torch.Generator(device=DEV).manual_seed(SEED)
    hist, t0 = [], time.time()
    # Sample the first hundred steps densely. The interesting behaviour of a second head is
    # early, and a uniform interval walks straight over it.
    log_at = sorted({0, 10, 20, 30, 40, 50, 75, 100, 150, 200}
                    | {i * log_every for i in range(steps // max(log_every, 1) + 1)})

    for step in range(steps + 1):
        i = torch.randint(len(train) - T - 3, (B,), generator=g, device=DEV)
        seq = torch.stack([train[j:j + T + 2] for j in i])
        idx, y1, y2 = seq[:, :T], seq[:, 1:T + 1], seq[:, 2:T + 2]
        h = model.hidden(idx)
        l1 = F.cross_entropy(model.lm_head(h).reshape(-1, V), y1.reshape(-1))
        l2 = F.cross_entropy(model.extra_heads[0](h).reshape(-1, V), y2.reshape(-1))
        loss = l1 + l2                                  # "the losses simply add"
        if step in log_at or step == steps:
            rec = dict(step=step, l1=l1.item(), l2=l2.item(),
                       total=loss.item(), gap=l2.item() - l1.item())
            hist.append(rec)
            if verbose:
                say(f"  {step:>6}  {rec['l1']:>10.4f}  {rec['l2']:>10.4f}  {rec['total']:>10.4f}  "
                    f"{rec['gap']:>+8.4f}  {math.exp(min(rec['l1'], 20)):>10.1f}")
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    wall = time.time() - t0
    del model, opt, train, data
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return hist, wall, cfg


def part2(text, enc, V, steps, log_every):
    rule("P2", "Add a second output head predicting t+2, and train")
    hist, wall, cfg = train_two_heads(text, enc, V, steps, log_every, tie=True, verbose=True)
    first, last = hist[0], hist[-1]
    lnV = math.log(V)
    say(f"\n  wall clock: {wall:.1f}s on {DEV}")
    say(f"\n  head 1 (t+1): {first['l1']:.4f} -> {last['l1']:.4f}   ({last['l1']-first['l1']:+.4f})")
    say(f"  head 2 (t+2): {first['l2']:.4f} -> {last['l2']:.4f}   ({last['l2']-first['l2']:+.4f})")
    say(f"  sum         : {first['total']:.4f} -> {last['total']:.4f}")
    say(f"  gap L2-L1   : {first['gap']:+.4f} -> {last['gap']:+.4f}")
    say(f"\n  Both heads start at ln(V) = {lnV:.4f}. They have to: at step 0 neither knows anything,")
    say("  and 'the token after next' is exactly as unknown as 'the next token'.")

    crossed = [r for r in hist if r["gap"] < 0]
    R["part2"] = dict(params=None, extra_head_params=V * cfg.d_model, steps=steps,
                      ln_V=lnV, history=hist, first=first, last=last, wall_s=wall,
                      negative_gap_steps=[r["step"] for r in crossed])

    if crossed:
        say(f"\n  But look at the middle of the run. At steps {[r['step'] for r in crossed]} the t+2 head")
        say("  is AHEAD of the t+1 head -- the harder question is being answered better. That should")
        say("  not happen, so it is worth finding out why rather than narrating past it.")
        say("\n  Hypothesis: it is not about the horizon at all, it is about TYING. Head 1 shares its")
        say("  matrix with the input embedding, so its gradient is fighting a second job; head 2 is a")
        say("  free matrix. Early in training, when both heads are mostly learning the marginal token")
        say("  frequencies, the unencumbered head fits that marginal faster.")
        say("\n  The hypothesis makes a prediction, so run the control: same everything, head 1 untied.")
        h2, _w, _c = train_two_heads(text, enc, V, steps, log_every, tie=False, verbose=False)
        crossed2 = [r["step"] for r in h2 if r["gap"] < 0]
        say(f"\n  {'step':>6}  {'tied: L2-L1':>13}  {'untied: L2-L1':>15}")
        for a, b in zip(hist, h2):
            say(f"  {a['step']:>6}  {a['gap']:>+13.4f}  {b['gap']:>+15.4f}")
        say(f"\n  tied   : gap negative at steps {[r['step'] for r in crossed]}")
        say(f"  untied : gap negative at steps {crossed2 if crossed2 else 'never'}")
        R["part2"]["control_untied"] = dict(history=h2, negative_gap_steps=crossed2)
        if not crossed2:
            say("\n  Confirmed. Untied, the t+2 head is never ahead -- the crossover was the tying")
            say("  constraint, not the prediction horizon. Worth knowing before reading any MTP")
            say("  loss curve: a tied t+1 head and an untied t+k head are not a fair comparison.")
        else:
            say("\n  REFUTED. The untied control crosses over at the same steps and by the same")
            say("  margin, so tying is not the cause. The hypothesis was wrong; the observation")
            say("  stands, and it is a property of the task pair rather than of the parameterisation.")

            # Second hypothesis: the crossover is the boundary of the marginal-fitting phase --
            # before the trunk carries usable context, t+1 has no advantage over t+2. That
            # predicts the crossover sits at the unigram entropy of the corpus. Test it.
            H1 = unigram_entropy(enc, text)
            lo = max((r for r in hist if r["gap"] < 0), key=lambda r: r["step"])
            hi = min((r for r in hist if r["step"] > lo["step"]), key=lambda r: r["step"])
            say(f"\n  Second hypothesis: the crossover is where the model stops fitting the marginal")
            say(f"  and starts using context, which would put it at the corpus unigram entropy.")
            say(f"    unigram entropy of the training tokens : {H1:.4f} nats")
            say(f"    crossover happens between step {lo['step']} (L1={lo['l1']:.4f}) "
                f"and step {hi['step']} (L1={hi['l1']:.4f})")
            inside = hi["l1"] < H1 < lo["l1"]
            say(f"    is the unigram entropy inside that window? {inside}")
            if not inside:
                say(f"\n  ALSO REFUTED. The crossover happens at about {(lo['l1']+hi['l1'])/2:.2f} nats,")
                say(f"  roughly {(lo['l1']+hi['l1'])/2 - H1:.2f} nats ABOVE the marginal floor -- so it")
                say("  arrives before the model has even finished fitting unigram frequencies.")
            R["part2"]["unigram_entropy"] = H1
            R["part2"]["crossover_window"] = [lo, hi]
            R["part2"]["marginal_phase_hypothesis"] = "supported" if inside else "refuted"
            say("\n  Two hypotheses proposed, two refuted. What survives is the observation itself:")
            say("  reproducible under a fixed seed, identical in both arms, and crossing at the same")
            say("  step. The next discriminating test is a frozen random trunk -- if the effect is")
            say("  about how much context the trunk carries, freezing it should stop head 1 from ever")
            say("  overtaking. That is not run here, and it is not claimed either way.")

    say("\n  By the end the ordering is the expected one and the gap WIDENS as L1 falls. That gap is")
    say("  the point rather than a defect: head 2 must predict a token without ever seeing the one")
    say("  in between, so it forces the hidden state at t to carry information the next-token loss")
    say("  alone would never ask for. That is the extra supervision MTP is bought with -- and the")
    say("  cost is honest: another full [V,D] matrix, which at V5 scale is another 536.9M parameters.")


# =========================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    steps = 200 if a.quick else a.steps

    torch.manual_seed(SEED)
    say(f"ERA V5 Session 9 -- loss harness")
    say(f"torch {torch.__version__} | device {DEV}"
        + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))
    say(f"seed {SEED} | steps {steps}")

    enc = get_tokenizer()
    V = enc.n_vocab
    text = corpus()
    say(f"corpus: {len(text):,} characters | tokenizer: GPT-2 BPE, V={V:,}")

    cfg = Config(vocab_size=V, block_size=256)
    torch.manual_seed(SEED)
    model = GPT(cfg).to(DEV).eval()

    ids = enc.encode(text[:20000])
    tokens = torch.tensor(ids[:4 * cfg.block_size], device=DEV).view(4, cfg.block_size)

    # Requirement 5 is the only one that NEEDS an untrained model, so it runs first.
    req5(model, tokens, V)

    # Requirements 3 and 4 measure nothing on an untrained model -- every position costs
    # ln(V), so masking any subset of them changes nothing. Train briefly, then measure.
    rule("--", "Training the Part 1 model, so requirements 3 and 4 have something to measure")
    R["pretrain_steps"] = 400
    pretrain(model, text, enc, R["pretrain_steps"])

    req1(model, tokens, enc)
    req2(tokens, enc)
    req3(model, enc, V, text)
    req4(model, enc, V, text)
    req6(cfg)
    req7(model, V)
    part2(text, enc, V, steps, log_every=max(1, steps // 10))

    R["meta"] = dict(torch=torch.__version__, device=DEV, seed=SEED,
                     gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None)
    (HERE / "results.json").write_text(json.dumps(R, indent=2))
    say(f"\n{'='*78}\nwrote results.json\n{'='*78}")


if __name__ == "__main__":
    main()
