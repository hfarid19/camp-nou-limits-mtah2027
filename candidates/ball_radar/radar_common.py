"""Ball Radar inputs and numpy inference (shared by the torch trainer and the pipeline).

The network (Kim et al., KDD 2023, adapted; see README.md) runs at STEP-frame resolution
(5 Hz) on canonical coordinates: x, y centred on the pitch and multiplied by the per-frame
attack sign, so the home team always attacks +x. Swapping home and away is then exactly
"negate every coordinate and the team flag", which is how training augments and how
inference symmetrises.

prepare(players, slot_team, ball, sign_frames, fps, frames) -> dict of model inputs at the
sampled frames:
    xs   (T, S, FS) float32 per-slot features (0 where the slot is empty)
    ms   (T, S)     float32 slot valid mask
    pos  (T, S, 2)  canonical slot positions / 10 m (0 where empty)
    xf   (T, FF)    float32 per-frame features (observed ball context)
forward(W, inp) -> dict(ball (T, 2) canonical / 10 m, g (T, S+1) player-level possession
probabilities (last column = dead / no player), team (T, 3) softmax H, A, D)
"""

import numpy as np

STEP = 5
FS = 11
FF = 10
L_HALF, W_HALF = 52.5, 34.0
SCALE = 10.0


def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def since_last(mask, cap):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(n) - last, cap).astype(np.float64)
    return np.minimum(out, cap)


def _mmean_axis0(v, hw):
    """Centred NaN-aware moving mean along axis 0 (any trailing shape)."""
    good = ~np.isnan(v)
    x = np.where(good, v, 0.0).astype(np.float64)
    c = good.astype(np.float64)
    pad = np.zeros((1,) + v.shape[1:])
    cx = np.concatenate((pad, np.cumsum(x, 0)))
    cc = np.concatenate((pad, np.cumsum(c, 0)))
    n = len(v)
    i = np.arange(n)
    lo, hi = np.clip(i - hw, 0, n), np.clip(i + hw + 1, 0, n)
    k = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(k > 0, (cx[hi] - cx[lo]) / np.maximum(k, 1), np.nan)
    return out


def canon(xy, sgn):
    """metres -> canonical / 10 m (home attacks +x). sgn broadcastable to xy[..., 0]."""
    out = np.empty(xy.shape, np.float64)
    out[..., 0] = (xy[..., 0] - L_HALF) * sgn / SCALE
    out[..., 1] = (xy[..., 1] - W_HALF) * sgn / SCALE
    return out


def uncanon(c, sgn):
    out = np.empty(c.shape, np.float64)
    out[..., 0] = c[..., 0] * SCALE * sgn + L_HALF
    out[..., 1] = c[..., 1] * SCALE * sgn + W_HALF
    return out


def interp_ball(ball):
    """Raw ball with every gap filled by linear interpolation (edges held); centre if none."""
    seen = ~np.isnan(ball[:, 0])
    out = np.empty(ball.shape, np.float64)
    if seen.sum() < 2:
        out[:] = (L_HALF, W_HALF)
        return out
    t = np.arange(len(ball))
    for i in (0, 1):
        out[:, i] = np.interp(t, t[seen], ball[seen, i].astype(np.float64))
    return out


def prepare(players, slot_team, ball, sign_frames, fps, frames):
    """Model inputs at the given frame indices (see module docstring)."""
    players = players.astype(np.float64)
    n, S = players.shape[:2]
    sg = np.asarray(sign_frames, np.float64)
    team = np.where(np.asarray(slot_team) == 0, 1.0, -1.0)
    # player velocity (m/s), jumps removed, +-2 frame smoothing, at full rate then sampled
    vel = np.full(players.shape, np.nan)
    vel[1:] = (players[1:] - players[:-1]) * fps
    spd = np.sqrt((vel ** 2).sum(-1))
    vel[spd > 12.0] = np.nan
    vel = _mmean_axis0(vel.reshape(n, -1), 2).reshape(players.shape)
    raw_seen = ~np.isnan(ball[:, 0])
    cap = 60 * fps
    since = since_last(raw_seen, cap)
    until = since_last(raw_seen[::-1], cap)[::-1]
    bvel = np.full(ball.shape, np.nan)
    bvel[1:] = (ball[1:] - ball[:-1]) * fps
    bspd = np.sqrt((bvel ** 2).sum(-1))
    bspd[bspd > 45] = np.nan
    bspd = _mmean_axis0(bspd, 2)
    mspd = np.nanmean(np.sqrt((vel ** 2).sum(-1)), 1)
    f = np.asarray(frames)
    s = sg[f]
    P = players[f]
    m = ~np.isnan(P[:, :, 0])
    pc = canon(P, s[:, None])
    v = vel[f] * s[:, None, None] / 5.0
    v = np.where(np.isnan(v), 0.0, v)
    seen = raw_seen[f]
    bi = interp_ball(ball)                                # linear interpolation over gaps
    bc = canon(bi[f], s)
    rel = bc[:, None, :] - pc
    dist = np.sqrt((rel ** 2).sum(-1))
    T = len(f)
    xs = np.zeros((T, S, FS), np.float64)
    xs[..., 0:2] = pc
    xs[..., 2:4] = v
    xs[..., 4] = team[None, :]
    xs[..., 5:7] = np.clip(rel, -3, 3)
    xs[..., 7] = np.minimum(dist, 3.0)
    xs[..., 8] = np.exp(-dist / 0.2)                      # ~1 within a couple of metres
    xs[..., 9] = seen[:, None]
    xs[..., 10] = np.sqrt((v ** 2).sum(-1))
    xs = np.where(np.isnan(xs), 0.0, xs)
    xs[~m] = 0.0
    pc = np.where(m[..., None], pc, 0.0)
    xf = np.zeros((T, FF), np.float64)
    xf[:, 0:2] = bc                                       # interpolated ball (residual base)
    xf[:, 2] = seen
    xf[:, 3] = np.log1p(since[f]) / 5.0
    xf[:, 4] = np.log1p(until[f]) / 5.0
    xf[:, 5] = m.sum(1) / 22.0
    xf[:, 6] = np.nan_to_num(bspd[f] / 20.0, nan=0.0)
    xf[:, 7] = np.nan_to_num(mspd[f] / 5.0, nan=0.0)
    xf[:, 8] = np.log1p(np.where(seen, 0, since[f] + until[f])) / 5.0
    xf[:, 9] = s * 0.0 + 1.0                              # bias
    return dict(xs=xs.astype(np.float32), ms=m.astype(np.float32), pos=pc.astype(np.float32),
                xf=xf.astype(np.float32))


def swap(inp):
    """Home <-> away: negate canonical coordinates and the team flag (labels H <-> A)."""
    xs = inp["xs"].copy()
    xs[..., 0:7] *= -1.0                 # pos, vel, team, rel
    xf = inp["xf"].copy()
    xf[:, 0:2] *= -1.0
    return dict(xs=xs, ms=inp["ms"], pos=-inp["pos"], xf=xf)


# ------------------------------------------------------------------------------------------
# numpy forward (mirrors train_radar.Radar exactly; float64 accumulate, deterministic)
# ------------------------------------------------------------------------------------------

def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _lin(W, k, x):
    return x @ W[k + ".weight"].T + W[k + ".bias"]


def _gru_dir(W, k, X, reverse):
    """X (T, B, I) -> (T, B, H) one direction of a torch GRU layer."""
    wi, wh = W[k + ".weight_ih"], W[k + ".weight_hh"]
    bi, bh = W[k + ".bias_ih"], W[k + ".bias_hh"]
    Hd = wh.shape[1]
    T, B = X.shape[:2]
    gi = X @ wi.T + bi                                    # (T, B, 3H)
    h = np.zeros((B, Hd))
    out = np.empty((T, B, Hd))
    order = range(T - 1, -1, -1) if reverse else range(T)
    whT = wh.T.copy()
    for t in order:
        gh = h @ whT + bh
        g = gi[t]
        r = _sig(g[:, :Hd] + gh[:, :Hd])
        z = _sig(g[:, Hd:2 * Hd] + gh[:, Hd:2 * Hd])
        nn_ = np.tanh(g[:, 2 * Hd:] + r * gh[:, 2 * Hd:])
        h = (1.0 - z) * nn_ + z * h
        out[t] = h
    return out


def _bigru(W, k, X):
    return np.concatenate([_gru_dir(W, k + ".f", X, False), _gru_dir(W, k + ".b", X, True)], -1)


def _layernorm(W, k, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * W[k + ".weight"] + W[k + ".bias"]


def _frame_stage(W, xs, ms, xf, heads):
    T, S, _ = xs.shape
    team = xs[..., 4]
    e = np.maximum(_lin(W, "phi1", xs), 0.0)
    e = np.maximum(_lin(W, "phi2", e), 0.0)                       # (T, S, D)
    D = e.shape[-1]
    # set attention block over all slots of a frame (permutation-equivariant)
    q, k_, v = _lin(W, "q", e), _lin(W, "k", e), _lin(W, "v", e)
    dh = D // heads
    q = q.reshape(T, S, heads, dh).transpose(0, 2, 1, 3)
    k_ = k_.reshape(T, S, heads, dh).transpose(0, 2, 1, 3)
    v = v.reshape(T, S, heads, dh).transpose(0, 2, 1, 3)
    sc = q @ k_.transpose(0, 1, 3, 2) / np.sqrt(dh)              # (T, h, S, S)
    sc = np.where(ms[:, None, None, :] > 0, sc, -1e4)
    sc = sc - sc.max(-1, keepdims=True)
    a = np.exp(sc)
    a = a / a.sum(-1, keepdims=True)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(T, S, D)
    e = _layernorm(W, "ln1", e + _lin(W, "o", o))
    e = _layernorm(W, "ln2", e + _lin(W, "ff2", np.maximum(_lin(W, "ff1", e), 0.0)))
    e = e * ms[..., None]
    ctx = _pool(e, ms, team)                                      # (T, 4D)
    ctx = np.maximum(_lin(W, "ctx", np.concatenate([ctx, xf], -1)), 0.0)
    return e, ctx


def forward(W, inp, heads=4, tchunk=2000, schunk=24):
    xs = inp["xs"].astype(np.float64)
    ms = inp["ms"].astype(np.float64)
    pos = inp["pos"].astype(np.float64)
    xf = inp["xf"].astype(np.float64)
    T, S, _ = xs.shape
    team = xs[..., 4]
    es, ctxs = [], []
    for t0 in range(0, T, tchunk):
        e, c = _frame_stage(W, xs[t0:t0 + tchunk], ms[t0:t0 + tchunk], xf[t0:t0 + tchunk], heads)
        es.append(e)
        ctxs.append(c)
    e, ctx = np.concatenate(es), np.concatenate(ctxs)
    # player-level possession classifier: slot-wise Bi-GRU over time (shared weights)
    hps = []
    for s0 in range(0, S, schunk):
        Xp = np.concatenate([e[:, s0:s0 + schunk],
                             np.broadcast_to(ctx[:, None, :], (T, min(schunk, S - s0), ctx.shape[-1]))], -1)
        hps.append(_bigru(W, "pgru", Xp))
    hp = np.concatenate(hps, 1)                                   # (T, S, 2Hp)
    lp = _lin(W, "plog", np.concatenate([hp, xs], -1))[..., 0]
    lp = np.where(ms > 0, lp, -1e4)
    hmax = np.where(ms[..., None] > 0, hp, -1e4).max(1)
    hmax = np.where(ms.sum(1, keepdims=True) > 0, hmax, 0.0)
    ld = _lin(W, "dlog", np.concatenate([hmax, ctx], -1))
    logits = np.concatenate([lp, ld], -1)
    logits = logits - logits.max(-1, keepdims=True)
    g = np.exp(logits)
    g = g / g.sum(-1, keepdims=True)                              # (T, S+1)
    # ball trajectory regressor: per-team pooling of [e, hp, g], expected possessor position
    gs = g[:, :S]
    xt = np.concatenate([e, hp, gs[..., None]], -1)
    xt = np.maximum(_lin(W, "bt1", xt), 0.0) * ms[..., None]
    z = _pool(xt, ms, team)
    epos = (gs[..., None] * pos).sum(1)
    gh = (gs * (team > 0)).sum(1, keepdims=True)
    ga = (gs * (team < 0)).sum(1, keepdims=True)
    zin = np.concatenate([z, epos, gh, ga, g[:, S:], ctx, xf], -1)
    zin = np.maximum(_lin(W, "bt2", zin), 0.0)
    hb = _bigru(W, "bgru", zin[:, None, :])[:, 0]                 # (T, 2Hb)
    ball = xf[:, 0:2] + _lin(W, "ball", hb)                       # residual on the interpolation
    tl = _lin(W, "team", hb)
    tl = tl - tl.max(-1, keepdims=True)
    tp = np.exp(tl)
    tp = tp / tp.sum(-1, keepdims=True)
    return dict(ball=ball, g=g, team=tp, epos=epos, gh=gh[:, 0], ga=ga[:, 0])


def _pool(e, ms, team):
    """masked mean and max over home slots and over away slots -> (T, 4D)."""
    outs = []
    for side in (team > 0, team < 0):
        mk = ms * side
        cnt = mk.sum(1, keepdims=True)
        mean = (e * mk[..., None]).sum(1) / np.maximum(cnt, 1.0)
        mx = np.where(mk[..., None] > 0, e, -1e4).max(1)
        mx = np.where(cnt > 0, mx, 0.0)
        outs += [mean, mx]
    return np.concatenate(outs, -1)


def load_weights(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def frame_sign(n, sign_half1, blocks):
    sign = np.full(n, float(sign_half1))
    for half, first, last in blocks:
        if half == 2:
            sign[first:] = -float(sign_half1)
    return sign


def run(W, players, slot_team, ball, sign_half1, blocks, fps):
    """Ball Radar outputs at every frame of the graded blocks: the network runs every STEP
    frames, outputs are linearly interpolated, and home / away are symmetrised by averaging with
    the swapped run. -> dict(team (N, 3) P(H, A, D), ball (N, 2) metres (NaN outside the
    blocks), g_dead, poss_h, poss_a (player-level possession mass of the dead class / per team))."""
    n = len(ball)
    sf = frame_sign(n, sign_half1, blocks)
    team = np.full((n, 3), 1.0 / 3)
    bxy = np.full((n, 2), np.nan)
    extra = np.zeros((n, 3))
    for _, first, last in blocks:
        frames = np.arange(first, last + 1, STEP)
        if frames[-1] != last:
            frames = np.append(frames, last)
        used = ~np.isnan(players[first:last + 1, :, 0]).all(0)
        inp = prepare(players[:, used], np.asarray(slot_team)[used], ball, sf, fps, frames)
        o = forward(W, inp)
        s = forward(W, swap(inp))
        tp = 0.5 * (o["team"] + s["team"][:, [1, 0, 2]])
        bc = 0.5 * (o["ball"] - s["ball"])
        ex = np.stack([0.5 * (o["g"][:, -1] + s["g"][:, -1]), 0.5 * (o["gh"] + s["ga"]),
                       0.5 * (o["ga"] + s["gh"])], 1)
        bm = uncanon(bc, sf[frames])
        t = np.arange(first, last + 1)
        for i in range(3):
            team[first:last + 1, i] = np.interp(t, frames, tp[:, i])
            extra[first:last + 1, i] = np.interp(t, frames, ex[:, i])
        for i in range(2):
            bxy[first:last + 1, i] = np.interp(t, frames, bm[:, i])
    team = team / team.sum(1, keepdims=True)
    return dict(team=team, ball=bxy, g_dead=extra[:, 0], poss_h=extra[:, 1], poss_a=extra[:, 2])
