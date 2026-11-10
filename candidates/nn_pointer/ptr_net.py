"""Possessor-pointer network (round 5, R5-3), torch side. ptr_np.py is the numpy mirror used at inference.

Inputs per 5 fps step t (ball_radar_pff/radar_common.prepare on the degraded tracks, plus nn_pff's frame
features):
    xs  (B, T, S, FS) per-slot features (canonical position, velocity, team flag +-1, offset to the
        interpolated ball, distance, ball-seen flag, speed); 0 where the slot is empty
    ms  (B, T, S)     slot mask
    xf  (B, T, FF)    per-frame ball context (radar_common)
    X   (B, K, T)     nn_pff features (features.compute, normalised, NaN -> 0)
Network:
    slot MLP -> nsab set-attention blocks over the slots of each frame (permutation-equivariant, no
    identities) -> per-team mean / max pooling -> [pool, xf, X] -> 1x1 conv -> dilated residual TCN over
    time (MS-TCN layers, L layers, dilation 2^i) = frame context c (B, C, T)
    per slot: [slot embedding, c] -> Dp -> Lp dilated residual convs over time (shared over slots)
    pointer logit per slot from [slot state, xs]; dead logit from [c, max over slots]
    -> g = softmax over S slots + dead.  Team = the possessor's team: P(H) = sum of g over home slots,
    P(A) over away slots, P(D) = g_dead.
    Extra heads on c: hand-over logit (as the TCN member's, for the LDE decoder) and a direct 3-class team
    head (auxiliary loss; mixed into the output only if cfg mix > 0).
Team swap = negate the canonical coordinates and the team flag (radar_common.swap) + nn_pff's feature
permutation; labels H <-> A; the pointer target (a slot) is unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FS, FF = 11, 10


class SAB(nn.Module):
    def __init__(self, D, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff1, self.ff2 = nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def forward(self, e, ms):                         # e (N, S, D), ms (N, S)
        N, S, D = e.shape
        h, dh = self.heads, D // self.heads
        q = self.q(e).view(N, S, h, dh).transpose(1, 2)
        k = self.k(e).view(N, S, h, dh).transpose(1, 2)
        v = self.v(e).view(N, S, h, dh).transpose(1, 2)
        sc = q @ k.transpose(-1, -2) / np.sqrt(dh)
        sc = sc.masked_fill(ms[:, None, None, :] <= 0, -1e4)
        o = (torch.softmax(sc, -1) @ v).transpose(1, 2).reshape(N, S, D)
        e = self.ln1(e + self.o(o))
        e = self.ln2(e + self.ff2(F.relu(self.ff1(e))))
        return e * ms[..., None]


class DRL(nn.Module):
    def __init__(self, d, c, p):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.out = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.out(F.relu(self.conv(x))))


def pool(e, ms, team):
    """e (N, S, D), ms (N, S), team (N, S) +-1 -> (N, 4D): masked mean / max over home, then away."""
    outs = []
    for side in (team > 0, team < 0):
        mk = ms * side.float()
        cnt = mk.sum(1, keepdim=True)
        mean = (e * mk[..., None]).sum(1) / cnt.clamp(min=1.0)
        mx = torch.where(mk[..., None] > 0, e, torch.full_like(e, -1e4)).max(1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        outs += [mean, mx]
    return torch.cat(outs, -1)


class PointerNet(nn.Module):
    def __init__(self, K, cfg):
        super().__init__()
        D, C, Dp = cfg["D"], cfg["C"], cfg["Dp"]
        fs, ff = cfg.get("FS", FS), cfg.get("FF", FF)        # ball-centred variant (ballc.py): 10 / 14
        self.phi1, self.phi2 = nn.Linear(fs, D), nn.Linear(D, D)
        self.sabs = nn.ModuleList([SAB(D, cfg["heads"]) for _ in range(cfg["nsab"])])
        self.in_drop = nn.Dropout(cfg["in_drop"])
        self.fin = nn.Conv1d(4 * D + ff + K, C, 1)
        self.tcn = nn.ModuleList([DRL(2 ** i, C, cfg["dropout"]) for i in range(cfg["L"])])
        self.sin = nn.Linear(D + C, Dp)
        self.stcn = nn.ModuleList([DRL(2 ** i, Dp, cfg["dropout_s"]) for i in range(cfg["Lp"])])
        self.plog = nn.Linear(Dp + fs, 1)
        self.dlog = nn.Linear(C + Dp, 1)
        self.ho = nn.Conv1d(C, 1, 1)
        self.dir = nn.Conv1d(C, 3, 1)
        self.noX = bool(cfg.get("noX", 0))          # 1: ignore nn_pff's frame features (tracks only)

    def forward(self, xs, ms, xf, X):
        B, T, S, _ = xs.shape
        team = xs[..., 4]
        e = F.relu(self.phi2(F.relu(self.phi1(xs)))).reshape(B * T, S, -1)
        msf = ms.reshape(B * T, S)
        for sab in self.sabs:
            e = sab(e, msf)
        D = e.shape[-1]
        pl = pool(e, msf, team.reshape(B * T, S)).view(B, T, 4 * D)
        if self.noX:
            X = torch.zeros_like(X)
        h = torch.cat([pl, xf, self.in_drop(X.transpose(1, 2))], -1).transpose(1, 2)   # (B, ., T)
        c = self.fin(h)
        for l in self.tcn:
            c = l(c)
        ct = c.transpose(1, 2)                                                       # (B, T, C)
        u = self.sin(torch.cat([e.view(B, T, S, D), ct[:, :, None, :].expand(B, T, S, ct.shape[-1])], -1))
        u = F.relu(u) * ms[..., None]
        u = u.permute(0, 2, 3, 1).reshape(B * S, -1, T)                               # (B*S, Dp, T)
        for l in self.stcn:
            u = l(u)
        u = u.view(B, S, -1, T).permute(0, 3, 1, 2)                                   # (B, T, S, Dp)
        lp = self.plog(torch.cat([u, xs], -1))[..., 0].masked_fill(ms <= 0, -1e4)
        umax = torch.where(ms[..., None] > 0, u, torch.full_like(u, -1e4)).max(2).values
        umax = torch.where(ms.sum(2, keepdim=True) > 0, umax, torch.zeros_like(umax))
        ld = self.dlog(torch.cat([ct, umax], -1))
        logits = torch.cat([lp, ld], -1)                                              # (B, T, S+1)
        return dict(logits=logits, ho=self.ho(c)[:, 0], dir=self.dir(c), team=team)


def team_probs(logits, team, ms):
    """pointer softmax -> (B, T, 3) log P(H), log P(A), log P(D) (log-sum-exp over each team's slots)."""
    S = team.shape[-1]
    lsm = F.log_softmax(logits, -1)
    neg = torch.full_like(lsm[..., :S], -1e4)
    lh = torch.logsumexp(torch.where((team > 0) & (ms > 0), lsm[..., :S], neg), -1)
    la = torch.logsumexp(torch.where((team < 0) & (ms > 0), lsm[..., :S], neg), -1)
    return torch.stack([lh, la, lsm[..., S]], -1), lsm
