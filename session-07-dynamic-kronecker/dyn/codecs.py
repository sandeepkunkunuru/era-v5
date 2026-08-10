"""Byte codecs: Kronecker v1 (the shipped baseline) and the dynamic-window proposals.

Every codec turns a token's UTF-8 bytes into a fixed-width code vector. The code is
never trained; a single shared `Linear(code_dim, d_model)` on top of it is the only
trainable object in the input path. That contract is identical across all codecs here,
so swapping one for another is a controlled experiment.

The baseline, as shipped and as taught in Session 7:

    kappa(t) = z( (1/sqrt(L)) * sum_p  onehot_256(byte_p) (x) onehot_P(p) ),  L = min(len, P)

with P = pos_dim = 32. Two properties follow from `onehot_P` and both are defects:

  * bytes at p >= P are DROPPED — a token longer than P bytes is silently cropped;
  * the code is 256*P wide no matter how short the token is — "apple" and "a" both
    pay for 32 position slots.

The fix is the one the field already applied to positional encoding (report §2j):
replace the *stored* one-hot over a bounded index with a *computed* basis defined for
every p. `fourier` and `randproj` below do exactly that, which removes the cap and lets
K < P shrink the code at the same time.
"""
import math

import torch

BYTE_VALUES = 256


# --------------------------------------------------------------------------- bases

def phi_onehot(positions: torch.Tensor, K: int) -> torch.Tensor:
    """v1's position factor: a one-hot over absolute position, undefined past K."""
    out = torch.zeros(*positions.shape, K, dtype=torch.float32)
    valid = positions < K
    idx = positions.clamp(max=K - 1)
    out.scatter_(-1, idx.unsqueeze(-1), 1.0)
    return out * valid.unsqueeze(-1)


def phi_fourier(positions: torch.Tensor, K: int, base: float = 1000.0) -> torch.Tensor:
    """Sinusoidal basis at geometric frequencies — the sinusoidal PE construction,
    reused as a *byte-position* factor. Defined for every p, so nothing is ever
    cropped, and smooth in p so shared prefixes stay close."""
    assert K % 2 == 0, "fourier basis needs an even K"
    half = K // 2
    p = positions.float().unsqueeze(-1)
    freqs = torch.exp(-math.log(base) * torch.arange(half, dtype=torch.float32) / half)
    ang = p * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1) * math.sqrt(2.0 / K)


def phi_randproj(positions: torch.Tensor, K: int, seed: int = 7) -> torch.Tensor:
    """A fixed pseudo-random unit vector per position. Near-orthogonal for distinct
    positions, so it is the collision-resistance ceiling — but it has no smoothness,
    so prefix similarity is destroyed. Included as a deliberate contrast to fourier."""
    pmax = int(positions.max().item()) + 1 if positions.numel() else 1
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(max(pmax, 1), K, generator=g)
    table = table / table.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return table[positions]


def phi_relative(positions: torch.Tensor, lengths: torch.Tensor, K: int,
                 base: float = 1000.0) -> torch.Tensor:
    """Fourier over *normalised* position u = p/(L-1), so a short token spreads across
    the whole basis instead of occupying a corner of it. This is the most literal
    reading of "don't waste 32 slots on 'a'" — and §Findings shows what it costs."""
    assert K % 2 == 0
    half = K // 2
    denom = (lengths - 1).clamp_min(1).float()
    u = positions.float() / denom
    scale = torch.exp(-math.log(base) * torch.arange(half, dtype=torch.float32) / half)
    ang = (u * (2 * math.pi)).unsqueeze(-1) * (1.0 / scale.clamp_min(1e-6))
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1) * math.sqrt(2.0 / K)


# --------------------------------------------------------------------------- codec

def phi_hybrid(positions: torch.Tensor, K: int, seed: int = 7) -> torch.Tensor:
    """Half smooth sinusoid, half fixed-random. The sinusoidal half keeps shared
    prefixes close (Kronecker's stated benefit); the random half supplies the
    near-orthogonality that pushes hard pairs apart. This is the recommended codec."""
    assert K % 4 == 0, "hybrid splits K in two even halves"
    half = K // 2
    return torch.cat([phi_fourier(positions, half),
                      phi_randproj(positions, half, seed) * math.sqrt(1.0 / half)],
                     dim=-1)


class ByteCodec:
    """Turns token strings into fixed-width, never-trained code vectors.

    kind:
      'onehot'   — Kronecker v1. Truncates at K bytes. code_dim = 256*K.
      'fourier'  — absolute position on a sinusoidal basis. No cap. code_dim = 256*K.
      'randproj' — absolute position, random orthogonal-ish basis. No cap.
      'relative' — position normalised by token length. No cap.
      'hybrid'   — fourier ⊕ randproj. No cap. The recommended codec.

    norm: '1/sqrt(L)' (the written brief) or '1/L' (what the lecture says at
    [01:50:54]). The report flags this discrepancy; `census.py` measures it.
    """

    def __init__(self, kind: str = "onehot", K: int = 32,
                 norm: str = "1/sqrt(L)", znorm: bool = True, seed: int = 7):
        assert kind in ("onehot", "fourier", "randproj", "relative", "hybrid")
        assert norm in ("1/sqrt(L)", "1/L")
        self.kind, self.K, self.norm, self.znorm, self.seed = kind, K, norm, znorm, seed

    @property
    def code_dim(self) -> int:
        return BYTE_VALUES * self.K

    @property
    def caps_length(self) -> bool:
        """True if the codec silently drops bytes past its window."""
        return self.kind == "onehot"

    def visible_bytes(self, raw: bytes) -> bytes:
        """The prefix of `raw` this codec can actually see."""
        return raw[: self.K] if self.caps_length else raw

    def encode(self, tokens: list[str]) -> torch.Tensor:
        """[N, code_dim] float32. Deterministic: same tokens -> same codes, always."""
        raws = [t.encode("utf-8") for t in tokens]
        vis = [self.visible_bytes(r) for r in raws]
        n = len(tokens)
        maxlen = max((len(v) for v in vis), default=0)
        code = torch.zeros(n, BYTE_VALUES, self.K, dtype=torch.float32)
        if maxlen == 0:
            return code.reshape(n, -1)

        lengths = torch.tensor([max(len(v), 1) for v in vis])
        # Pad byte values into a rectangle so the position loop can be vectorised.
        bytes_mat = torch.zeros(n, maxlen, dtype=torch.long)
        valid = torch.zeros(n, maxlen, dtype=torch.bool)
        for i, v in enumerate(vis):
            if v:
                bytes_mat[i, : len(v)] = torch.tensor(list(v), dtype=torch.long)
                valid[i, : len(v)] = True

        pos = torch.arange(maxlen).unsqueeze(0).expand(n, -1)
        if self.kind == "onehot":
            basis = phi_onehot(pos, self.K)
        elif self.kind == "fourier":
            basis = phi_fourier(pos, self.K)
        elif self.kind == "randproj":
            basis = phi_randproj(pos, self.K, self.seed)
        elif self.kind == "hybrid":
            basis = phi_hybrid(pos, self.K, self.seed)
        else:
            basis = phi_relative(pos, lengths.unsqueeze(-1).expand(-1, maxlen), self.K)
        basis = basis * valid.unsqueeze(-1)                      # zero out padding

        # code[i, b, :] = sum over positions p whose byte is b, of basis[i, p, :]
        rows = torch.arange(n).unsqueeze(-1).expand(-1, maxlen)
        code.index_put_((rows.reshape(-1), bytes_mat.reshape(-1)),
                        basis.reshape(-1, self.K), accumulate=True)

        denom = lengths.float()
        scale = denom.rsqrt() if self.norm == "1/sqrt(L)" else 1.0 / denom
        code = code * scale.view(-1, 1, 1)
        code = code.reshape(n, -1)
        if self.znorm:
            code = (code - code.mean(-1, keepdim=True)) / code.std(-1, keepdim=True).clamp_min(1e-6)
        return code

    def __repr__(self) -> str:
        cap = f"caps@{self.K}B" if self.caps_length else "no cap"
        return f"ByteCodec({self.kind}, K={self.K}, dim={self.code_dim}, {cap})"


def parameter_count(codec: ByteCodec, d_model: int) -> int:
    """Trainable parameters in the input path: the single shared projection."""
    return codec.code_dim * d_model
