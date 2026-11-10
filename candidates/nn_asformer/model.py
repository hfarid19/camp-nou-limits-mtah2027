"""Torch models for nn_asformer (training only; inference is asf_np.py, plain numpy).

Three temporal segmentation networks share one skeleton: an encoder stage on the 5 fps features and
S refinement (decoder) stages, each stage a stack of pre-norm residual blocks and two linear heads
(3-class H/A/D logits, 2 boundary logits: H<->A, alive<->dead), exactly the outputs the MS-TCN
member produces so its ASRF decoder can be reused.

Block (pre-norm, GELU, residual, dropout, stochastic depth), layer i of a stage:
    h = x + DropPath(ConvFF(LN(x)))        ConvFF: dilated conv k=3 dilation 2^i -> GELU -> 1x1
    h = h + DropPath(alpha * Mixer(LN(h)))  alpha: learnable scalar gate (ASFormer), init 1
Mixer:
    arch "asformer"  local windowed multi-head self-attention, half-width w_i = min(2^i, w_max)
                     (window doubling per layer as in ASFormer, Yi et al. BMVC 2021), with a learned
                     relative position bias per head over -w..w (a relative, window-local encoding:
                     crops and full halves see the same thing). Decoder stages use cross-attention
                     (q from the decoder, k/v from the encoder output), same window.
    arch "global"    full self-attention over the whole sequence with ALiBi (linear distance bias,
                     fixed per-head slopes), torch.nn.functional.scaled_dot_product_attention.
                     Decoder stages: global cross-attention.
    arch "temba"     bidirectional Mamba-2 style selective state-space mixer (SSD, scalar decay per
                     head; Dao & Gu 2024) after the dilated conv, in the spirit of MS-Temba
                     (arXiv 2501.06138: dilated multi-scale temporal convs + Mamba). Written from the
                     papers; no code copied. Decoder stages = the same blocks on the softmax outputs.
Decoder input = softmax(class logits) ++ sigmoid(boundary logits) of the previous stage (5 ch).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DIL_CYCLE = 12   # layer i uses dilation 2^(i mod 12) (2^11 = 2048 steps = the crop length)

CFG = dict(arch="asformer", d=64, L=9, S=2, heads=4, w_max=256, dropout=0.3, drop_path=0.1, in_drop=0.0,
           d_dec=0, L_dec=0,            # 0: same as encoder
           ssm_N=16, ssm_expand=2, ssm_chunk=64,
           ln_eps=1e-5)


def gelu(x):
    return F.gelu(x, approximate="tanh")


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p
        return x * keep / (1 - self.p)


class ConvFF(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.out = nn.Linear(d, d)

    def forward(self, x):            # (B, T, d)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out(gelu(h))


def band_attention(q, k, v, w, bias, T):
    """Exact banded attention |t - s| <= w, computed on query chunks. q, k, v (B, h, Tq, dh) with
    Tq = Tk = T (self or cross with equal length). bias (h, 2w+1) relative position bias."""
    B, h, _, dh = q.shape
    c = max(w, 16)
    Tp = ((T + c - 1) // c) * c
    pad = Tp - T
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    nc = Tp // c
    qc = q.reshape(B, h, nc, c, dh)
    kp = F.pad(k, (0, 0, w, w))
    vp = F.pad(v, (0, 0, w, w))
    kw = kp.unfold(2, c + 2 * w, c).transpose(-1, -2)      # (B, h, nc, c+2w, dh)
    vw = vp.unfold(2, c + 2 * w, c).transpose(-1, -2)
    s = torch.matmul(qc, kw.transpose(-1, -2)) / math.sqrt(dh)   # (B, h, nc, c, c+2w)
    i = torch.arange(c, device=q.device)[:, None]
    j = torch.arange(c + 2 * w, device=q.device)[None, :]
    rel = j - w - i                                              # (c, c+2w)
    valid = rel.abs() <= w
    kabs = torch.arange(nc, device=q.device)[:, None, None] * c - w + j[None]   # (nc, 1, c+2w)
    valid = valid[None] & (kabs >= 0) & (kabs < T)
    s = s + bias[:, rel.clamp(-w, w) + w][None, :, None]           # (1, h, 1, c, c+2w)
    s = s.masked_fill(~valid[None, None], -1e9)
    p = torch.softmax(s, -1)
    out = torch.matmul(p, vw)                                    # (B, h, nc, c, dh)
    return out.reshape(B, h, Tp, dh)[:, :, :T]


def alibi_slopes(h):
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / h) for i in range(h)])


class Attention(nn.Module):
    def __init__(self, d, heads, w, cross=False, glob=False):
        super().__init__()
        self.h, self.dh, self.w, self.cross, self.glob = heads, d // heads, w, cross, glob
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        if glob:
            self.register_buffer("slopes", alibi_slopes(heads))
        else:
            self.bias = nn.Parameter(torch.zeros(heads, 2 * w + 1))

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, x, f=None):
        src = f if self.cross else x
        q, k, v = self._split(self.q(x)), self._split(self.k(src)), self._split(self.v(src))
        T = x.shape[1]
        if self.glob:
            pos = torch.arange(T, device=x.device)
            dist = (pos[None, :] - pos[:, None]).abs().float()
            mask = -self.slopes[:, None, None] * dist[None]       # (h, T, T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None])
        else:
            out = band_attention(q, k, v, self.w, self.bias, T)
        return self.o(out.transpose(1, 2).reshape(x.shape[0], T, -1))


def segsum_exp(a):
    """a (..., L) per-step log decays -> M[t, s] = exp(sum_{r=s+1..t} a_r) for s <= t else 0."""
    L = a.shape[-1]
    cs = torch.cumsum(a, -1)
    diff = cs[..., :, None] - cs[..., None, :]                   # cs_t - cs_s
    tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=a.device))
    return torch.exp(diff.masked_fill(~tri, -1e9)) * tri


def ssd_scan(x, dt, A, Bm, Cm, chunk):
    """Selective state space (Mamba-2 SSD, scalar decay per head), causal.
    x (B, T, H, P), dt (B, T, H) > 0, A (H,) < 0, Bm / Cm (B, T, H, N). -> y (B, T, H, P)."""
    B, T, H, P = x.shape
    N = Bm.shape[-1]
    L = chunk
    Tp = ((T + L - 1) // L) * L
    if Tp != T:
        pd = Tp - T
        x, dt, Bm, Cm = (F.pad(t, (0, 0) * (t.dim() - 2) + (0, pd)) if t.dim() > 3 else F.pad(t, (0, 0, 0, pd))
                         for t in (x, dt, Bm, Cm))
    nc = Tp // L
    x = x.reshape(B, nc, L, H, P)
    dt = dt.reshape(B, nc, L, H)
    Bm = Bm.reshape(B, nc, L, H, N)
    Cm = Cm.reshape(B, nc, L, H, N)
    a = dt * A                                                   # (B, nc, L, H) log decay per step
    a = a.permute(0, 3, 1, 2)                                    # (B, H, nc, L)
    M = segsum_exp(a)                                            # (B, H, nc, L, L)
    cs = torch.cumsum(a, -1)                                     # (B, H, nc, L)
    xd = x * dt[..., None]                                       # (B, nc, L, H, P)
    # diagonal (within chunk): y_t = sum_s C_t.B_s M[t,s] xd_s
    CB = torch.einsum("bclhn,bcshn->bhcls", Cm, Bm) * M
    y = torch.einsum("bhcls,bcshp->bclhp", CB, xd)
    # chunk end states: S_c = sum_s exp(cs_L - cs_s) B_s^T xd_s
    dec_end = torch.exp(cs[..., -1:] - cs)                       # (B, H, nc, L)
    st = torch.einsum("bhcl,bclhn,bclhp->bchnp", dec_end, Bm, xd)   # (B, nc, H, N, P)
    tot = torch.exp(cs[..., -1])                                 # (B, H, nc)
    S = torch.zeros(B, H, N, P, device=x.device, dtype=x.dtype)
    states = []
    for c in range(nc):
        states.append(S)
        S = tot[:, :, c, None, None] * S + st[:, c]
    Sprev = torch.stack(states, 1)                               # (B, nc, H, N, P) state entering chunk
    dec_in = torch.exp(cs)                                       # (B, H, nc, L)
    y = y + torch.einsum("bhcl,bclhn,bchnp->bclhp", dec_in, Cm, Sprev)
    return y.reshape(B, Tp, H, P)[:, :T]


class SSM(nn.Module):
    """Bidirectional Mamba-2 style mixer: two SSD scans (forward, backward) with their own B, C, dt
    and A, sharing the input projection; gated output."""

    def __init__(self, d, heads, N, expand, chunk):
        super().__init__()
        self.di = d * expand
        self.H, self.N, self.chunk = heads, N, chunk
        self.P = self.di // heads
        self.inp = nn.Linear(d, 2 * self.di + 2 * (2 * N + heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, heads + 1).float()).repeat(2).view(2, heads))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.rand(2, heads) * 0.09 + 0.01)))
        self.D = nn.Parameter(torch.ones(2, heads))
        self.out = nn.Linear(self.di, d)

    def forward(self, x):
        B, T, _ = x.shape
        z = self.inp(x)
        xs, g, rest = torch.split(z, [self.di, self.di, 2 * (2 * self.N + self.H)], -1)
        xs = xs.view(B, T, self.H, self.P)
        y = 0
        for k in range(2):
            r = rest[..., k * (2 * self.N + self.H):(k + 1) * (2 * self.N + self.H)]
            Bm, Cm, dt = torch.split(r, [self.N, self.N, self.H], -1)
            Bm = Bm[:, :, None, :].expand(B, T, self.H, self.N)
            Cm = Cm[:, :, None, :].expand(B, T, self.H, self.N)
            dt = F.softplus(dt + self.dt_bias[k])
            A = -torch.exp(self.A_log[k])
            if k == 1:
                xs_k, dt_k, Bm, Cm = xs.flip(1), dt.flip(1), Bm.flip(1), Cm.flip(1)
            else:
                xs_k, dt_k = xs, dt
            yk = ssd_scan(xs_k, dt_k, A, Bm, Cm, self.chunk) + xs_k * self.D[k][None, None, :, None]
            if k == 1:
                yk = yk.flip(1)
            y = y + yk
        y = y.reshape(B, T, self.di) * F.silu(g)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, cfg, d, dil, w, cross, dp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=cfg["ln_eps"])
        self.ff = ConvFF(d, dil)
        self.ln2 = nn.LayerNorm(d, eps=cfg["ln_eps"])
        arch = cfg["arch"]
        if arch == "temba":
            self.mix = SSM(d, cfg["heads"], cfg["ssm_N"], cfg["ssm_expand"], cfg["ssm_chunk"])
            self.cross = False
        else:
            self.mix = Attention(d, cfg["heads"], w, cross=cross, glob=(arch == "global"))
            self.cross = cross
        self.alpha = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(cfg["dropout"])
        self.dp = DropPath(dp)

    def forward(self, x, f=None):
        x = x + self.dp(self.drop(self.ff(self.ln1(x))))
        m = self.mix(self.ln2(x), f) if self.cross else self.mix(self.ln2(x))
        return x + self.dp(self.drop(self.alpha * m))


class Stage(nn.Module):
    def __init__(self, cfg, cin, d, L, cross):
        super().__init__()
        self.inp = nn.Linear(cin, d)
        n_all = cfg["L"] + cfg["S"] * (cfg["L_dec"] or cfg["L"])
        self.blocks = nn.ModuleList()
        for i in range(L):
            dil = 2 ** (i % DIL_CYCLE)              # dilation / window cycle every 12 layers (deep nets)
            w = min(dil, cfg["w_max"])
            dp = cfg["drop_path"] * (i + 1) / max(L, 1)
            self.blocks.append(Block(cfg, d, dil, w, cross, dp))
        self.ln = nn.LayerNorm(d, eps=cfg["ln_eps"])
        self.head_c = nn.Linear(d, 3)
        self.head_b = nn.Linear(d, 2)

    def forward(self, x, f=None):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, f)
        h = self.ln(h)
        return h, self.head_c(h), self.head_b(h)


class Net(nn.Module):
    def __init__(self, K, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_drop = nn.Dropout(cfg["in_drop"])
        self.enc = Stage(cfg, K, cfg["d"], cfg["L"], False)
        dd, Ld = cfg["d_dec"] or cfg["d"], cfg["L_dec"] or cfg["L"]
        cross = cfg["arch"] in ("asformer", "global")
        self.proj = nn.Linear(cfg["d"], dd) if (cross and dd != cfg["d"]) else None
        self.dec = nn.ModuleList([Stage(cfg, 5, dd, Ld, cross) for _ in range(cfg["S"])])

    def forward(self, x):
        """x (B, T, K) normalised features -> lists of class logits (B, T, 3) and boundary logits
        (B, T, 2), one per stage."""
        f, c, b = self.enc(self.in_drop(x))
        if self.proj is not None:
            f = self.proj(f)
        cs, bs = [c], [b]
        for st in self.dec:
            inp = torch.cat([torch.softmax(cs[-1], -1), torch.sigmoid(bs[-1])], -1)
            _, c, b = st(inp, f)
            cs.append(c)
            bs.append(b)
        return cs, bs


def n_params(m):
    return sum(p.numel() for p in m.parameters())
