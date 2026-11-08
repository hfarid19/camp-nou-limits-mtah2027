"""Vectorised pitch-control features (identity-free, team-relative, H/A swap-aware).

compute(players, slot_team, ball, sign, pitch, fps) -> (F float32 (N, K), names, kinds)

Two pitch-control models, both symmetric in the two teams (no attacker / defender roles: which
team attacks is exactly what we are trying to infer):

1. Spearman (2018), "Beyond Expected Goals" (MIT Sloan SAC): each player's time to intercept
   tti = t_react + |p + v * t_react - x| / v_max, a logistic intercept probability
   P_int(T) = 1 / (1 + exp(-pi / (sqrt(3) sigma) * (T - tti))), and the control ODE
   dPPCF_j / dT = (1 - sum_k PPCF_k) * P_int_j(T) * lambda, integrated from the ball's arrival
   time at x. Constants from the paper / Laurie Shaw's public implementation: t_react 0.7 s,
   v_max 5 m/s, sigma 0.45 s, lambda 4.3 /s (same for both teams), ball speed 15 m/s.
   The ODE is solved with an exponential integrator (exact for rates held constant over a step):
   remaining *= exp(-R dt), the drop shared in proportion to the rates, so a coarse dt = 0.08 s is
   stable. Only the K nearest-in-time players per team enter (Spearman also drops players whose
   tti is far above the minimum).
2. Fernandez & Bornn (2018), "Wide Open Spaces" (MIT Sloan SAC): each player's influence is a
   bivariate normal centred at p + 0.5 v, stretched along the velocity by the speed ratio
   (v / 13 m/s)^2, with a radius growing from 4 m to 10 m with the distance to the ball;
   control = sigmoid(sum_home I - sum_away I), I normalised to 1 at the centre.

Targets: the (observed or interpolated) ball, the ball's future positions (+0.4 / 1 / 2 s, with
that flight time, players at t: who will win the ball where it is going), a 5 m ring around the
ball, and a coarse grid (area share, every 5th frame, linearly interpolated).

kinds follow features.py: "anti" = negated when home/away are swapped, "H:x"/"A:x" partners.
"""

import numpy as np

T_REACT, V_MAX, TTI_SIGMA, LAMBDA, BALL_SPEED = 0.7, 5.0, 0.45, 4.3, 15.0
K_NEAR = 4            # players per team in the Spearman integral
DT, STEPS = 0.08, 36  # integration grid (2.9 s after the start)
SLOTS_PER_TEAM = 14   # compacted slots per team (ghosts / fragments beyond this are dropped)
CHUNK = 20000
CTRL_HI = 0.85        # "confident control" threshold of the Spearman control at the ball


# ------------------------------------------------------------------------------------------
# helpers (same conventions as learned_dfl_evidence/features.py)
# ------------------------------------------------------------------------------------------

def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def fill_gaps(values, max_gap):
    out = values.copy()
    good = ~np.isnan(values)
    if good.sum() < 2:
        return out
    idx = np.arange(len(values))
    interp = np.interp(idx, idx[good], values[good])
    for s, e in zip(*runs_of(~good)):
        if s > 0 and e < len(values) and e - s <= max_gap:
            out[s:e] = interp[s:e]
    return out


def mmean(values, lo_off, hi_off):
    values = np.asarray(values, np.float64)
    v = np.where(np.isnan(values), 0.0, values)
    c = (~np.isnan(values)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    n = len(values)
    i = np.arange(n)
    lo = np.clip(i + lo_off, 0, n)
    hi = np.clip(i + hi_off + 1, 0, n)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[hi] - cv[lo]) / np.maximum(cnt, 1), np.nan)


def mmean2d(values, hw):
    v = np.where(np.isnan(values), 0.0, values).astype(np.float64)
    c = (~np.isnan(values)).astype(np.float64)
    cv = np.concatenate((np.zeros((1, v.shape[1])), np.cumsum(v, 0)))
    cc = np.concatenate((np.zeros((1, v.shape[1])), np.cumsum(c, 0)))
    n = len(values)
    i = np.arange(n)
    lo, hi = np.clip(i - hw, 0, n), np.clip(i + hw + 1, 0, n)
    cnt = cc[hi] - cc[lo]
    out = np.where(cnt > 0, (cv[hi] - cv[lo]) / np.maximum(cnt, 1), np.nan)
    out[np.isnan(values)] = np.nan
    return out.astype(np.float32)


def shift(x, k):
    y = np.full_like(x, np.nan)
    if k > 0:
        y[:-k] = x[k:]
    elif k < 0:
        y[-k:] = x[:k]
    else:
        y[:] = x
    return y


def since_last(mask, cap):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(n) - last, cap).astype(np.float64)
    return np.minimum(out, cap)


def until_next(mask, cap):
    return since_last(mask[::-1], cap)[::-1]


# ------------------------------------------------------------------------------------------
# kinematics
# ------------------------------------------------------------------------------------------

def kinematics(players, slot_team, fps):
    """-> per team (pos (N, K, 2), vel (N, K, 2)) compacted to SLOTS_PER_TEAM, NaN-padded."""
    players = players.astype(np.float32)
    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * fps
    spd = np.sqrt((vel ** 2).sum(-1))
    vel[spd > 12.0] = np.nan
    vx = mmean2d(vel[:, :, 0], 2)
    vy = mmean2d(vel[:, :, 1], 2)
    v = np.stack([vx, vy], -1)
    v = np.where(np.isnan(v), 0.0, v).astype(np.float32)       # unknown velocity -> standing
    out = {}
    for tag, team in (("H", 0), ("A", 1)):
        cols = np.where(slot_team == team)[0]
        p = players[:, cols]
        vv = v[:, cols]
        miss = np.isnan(p[:, :, 0])
        order = np.argsort(miss, axis=1, kind="stable")[:, :SLOTS_PER_TEAM]
        k = order.shape[1]
        pk = np.take_along_axis(p, order[:, :, None], 1)
        vk = np.take_along_axis(vv, order[:, :, None], 1)
        if k < SLOTS_PER_TEAM:
            pad = SLOTS_PER_TEAM - k
            pk = np.concatenate([pk, np.full((len(p), pad, 2), np.nan, np.float32)], 1)
            vk = np.concatenate([vk, np.zeros((len(p), pad, 2), np.float32)], 1)
        out[tag] = (pk, vk)
    return out


# ------------------------------------------------------------------------------------------
# Spearman (2018) potential pitch control at one target per frame
# ------------------------------------------------------------------------------------------

def _tti(pos, vel, target):
    """pos/vel (C, K, 2), target (C, 2) -> tti (C, K), inf where the slot is empty."""
    r = pos + vel * T_REACT
    d = np.sqrt(((target[:, None, :] - r) ** 2).sum(-1))
    t = T_REACT + d / V_MAX
    return np.where(np.isnan(t), np.inf, t)


def spearman(kin, target, flight=None):
    """-> (pc_home (N,), tti_H sorted (N, 2), tti_A sorted (N, 2)); NaN where target is NaN."""
    n = len(target)
    pc = np.full(n, np.nan)
    tH = np.full((n, 2), np.nan)
    tA = np.full((n, 2), np.nan)
    ok_all = ~np.isnan(target[:, 0])
    c0 = np.pi / np.sqrt(3.0) / TTI_SIGMA
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        ok = ok_all[a:b]
        if not ok.any():
            continue
        rows = np.arange(a, b)[ok]
        tg = target[rows].astype(np.float64)
        th = np.sort(_tti(kin["H"][0][rows], kin["H"][1][rows], tg), 1)[:, :K_NEAR]
        ta = np.sort(_tti(kin["A"][0][rows], kin["A"][1][rows], tg), 1)[:, :K_NEAR]
        tH[rows] = np.where(np.isinf(th[:, :2]), np.nan, th[:, :2])
        tA[rows] = np.where(np.isinf(ta[:, :2]), np.nan, ta[:, :2])
        tt = np.concatenate([th, ta], 1)                      # (C, 2K)
        fl = np.zeros(len(rows)) if flight is None else flight[rows]
        tmin = np.min(tt, 1)
        t0 = np.maximum(fl, np.where(np.isfinite(tmin), tmin, 0.0) - 1.0)
        tt = np.where(np.isfinite(tt), tt, 1e3)
        rem = np.ones(len(rows))
        acc = np.zeros_like(tt)
        for s in range(STEPS):
            T = t0 + (s + 0.5) * DT
            rate = LAMBDA / (1.0 + np.exp(-c0 * (T[:, None] - tt)))
            R = rate.sum(1)
            new = rem * np.exp(-R * DT)
            drop = rem - new
            acc += (drop / np.maximum(R, 1e-12))[:, None] * rate
            rem = new
        h, w = acc[:, :K_NEAR].sum(1), acc[:, K_NEAR:].sum(1)
        tot = h + w
        pc[rows] = np.where(tot > 1e-9, h / np.maximum(tot, 1e-12), 0.5)
    return pc, tH, tA


# ------------------------------------------------------------------------------------------
# Fernandez & Bornn (2018) influence-based control at several targets per frame
# ------------------------------------------------------------------------------------------

def _fb_params(pos, vel, ball):
    """-> mu (C,K,2), inverse covariance (C,K,2,2) for every compacted slot."""
    mu = pos + 0.5 * vel
    speed = np.sqrt((vel ** 2).sum(-1))
    srat = np.minimum(speed ** 2 / 13.0 ** 2, 0.999)
    dball = np.sqrt(((pos - ball[:, None, :]) ** 2).sum(-1))
    dball = np.where(np.isnan(dball), 18.0, dball)
    R = np.minimum(4.0 + dball ** 3 / (18.0 ** 3 / 6.0), 10.0)
    sx = (R + R * srat) / 2.0
    sy = (R - R * srat) / 2.0
    th = np.arctan2(vel[..., 1], vel[..., 0])
    c, s = np.cos(th), np.sin(th)
    # Cov = Rot diag(sx^2, sy^2) Rot^T -> inverse = Rot diag(1/sx^2, 1/sy^2) Rot^T
    a, b = 1.0 / sx ** 2, 1.0 / sy ** 2
    i00 = c * c * a + s * s * b
    i11 = s * s * a + c * c * b
    i01 = c * s * (a - b)
    return mu, i00, i01, i11


def fb_control(kin, ball, targets):
    """ball (N, 2) (filled), targets (N, M, 2) -> home control (N, M) in [0, 1] (NaN where the
    target is NaN)."""
    n, m = targets.shape[:2]
    out = np.full((n, m), np.nan)
    bfill = np.where(np.isnan(ball), np.array([52.5, 34.0]), ball)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        tot = np.zeros((b - a, m))
        for tag, sgn in (("H", 1.0), ("A", -1.0)):
            pos, vel = kin[tag][0][a:b].astype(np.float64), kin[tag][1][a:b].astype(np.float64)
            mu, i00, i01, i11 = _fb_params(pos, vel, bfill[a:b])
            dx = targets[a:b, :, None, 0] - mu[:, None, :, 0]      # (C, M, K)
            dy = targets[a:b, :, None, 1] - mu[:, None, :, 1]
            q = i00[:, None] * dx * dx + 2 * i01[:, None] * dx * dy + i11[:, None] * dy * dy
            infl = np.exp(-0.5 * q)
            tot += sgn * np.nansum(infl, 2)
        out[a:b] = 1.0 / (1.0 + np.exp(-tot))
    out[np.isnan(targets[:, :, 0])] = np.nan
    return out


# ------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------

def compute(players, slot_team, ball_raw, sign, pitch, fps):
    L, W = pitch
    n = len(ball_raw)
    cols, names, kinds = [], [], []

    def add(name, values, kind="sym"):
        cols.append(np.asarray(values, np.float32))
        names.append(name)
        kinds.append(kind)

    kin = kinematics(players, slot_team, fps)
    raw_seen = ~np.isnan(ball_raw[:, 0])
    b64 = ball_raw.astype(np.float64)
    ball_s = np.stack([fill_gaps(b64[:, 0], int(0.5 * fps)), fill_gaps(b64[:, 1], int(0.5 * fps))], 1)
    ball_i = np.stack([fill_gaps(b64[:, 0], int(3.0 * fps)), fill_gaps(b64[:, 1], int(3.0 * fps))], 1)

    # ---- Spearman at the ball (short-filled and 3 s-interpolated)
    pc_s, tH, tA = spearman(kin, ball_s)
    pc_i, tHi, tAi = spearman(kin, ball_i)
    add("pcs_ball", pc_s - 0.5, "anti")
    add("pcs_ball_i", pc_i - 0.5, "anti")
    add("H_tti1", tHi[:, 0], "H:tti1")
    add("A_tti1", tAi[:, 0], "A:tti1")
    add("H_tti2", tHi[:, 1], "H:tti2")
    add("A_tti2", tAi[:, 1], "A:tti2")
    tti_d = tHi[:, 0] - tAi[:, 0]
    add("tti_d1", tti_d, "anti")
    add("tti_min", np.fmin(tHi[:, 0], tAi[:, 0]))
    add("tti_gap12", np.fmin(tHi[:, 1], tAi[:, 1]) - np.fmin(tHi[:, 0], tAi[:, 0]))

    # ---- Spearman where the ball is going (players at t, flight time = real time ahead)
    for sec in (0.4, 1.0, 2.0):
        k = int(round(sec * fps))
        tgt = np.stack([shift(ball_i[:, 0], k), shift(ball_i[:, 1], k)], 1)
        pcf, _, _ = spearman(kin, tgt, flight=np.full(n, sec))
        add("pcs_fut{}".format(sec), pcf - 0.5, "anti")

    # ---- Fernandez & Bornn at the ball and on a 5 m ring around it
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = ball_i[:, None, :] + 5.0 * np.stack([np.cos(ang), np.sin(ang)], 1)[None]
    tg = np.concatenate([ball_i[:, None, :], ring], 1)
    fb = fb_control(kin, ball_i, tg)
    add("pcf_ball", fb[:, 0] - 0.5, "anti")
    add("pcf_ring", np.nanmean(fb[:, 1:], 1) - 0.5, "anti")
    # ring points in the direction home attacks minus the ones behind (who controls ahead)
    fwd = np.cos(ang)[None] * sign[:, None]
    add("pcf_ring_fwd", np.nansum((fb[:, 1:] - 0.5) * fwd, 1) / 4.0, "anti")

    # ---- area share on a coarse grid (every 5th frame, interpolated)
    gx, gy = np.meshgrid(np.linspace(5, L - 5, 11), np.linspace(4, W - 4, 7))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    sub = np.arange(0, n, 5)
    kin_sub = {t: (kin[t][0][sub], kin[t][1][sub]) for t in kin}
    tg = np.broadcast_to(grid[None], (len(sub), len(grid), 2)).copy()
    g = fb_control(kin_sub, ball_i[sub], tg)
    area = np.interp(np.arange(n), sub, g.mean(1))
    dgb = np.sqrt(((grid[None] - ball_i[sub][:, None, :]) ** 2).sum(-1))
    near = np.where(dgb < 20.0, g, np.nan)
    with np.errstate(all="ignore"):
        an = np.nanmean(near, 1)
    good = ~np.isnan(an)
    area_near = np.interp(np.arange(n), sub[good], an[good]) if good.sum() > 1 else np.full(n, 0.5)
    area_near[np.isnan(ball_i[:, 0])] = np.nan
    # home's control of the half it attacks (attacking-space share)
    xside = (grid[None, :, 0] - L / 2) * sign[sub][:, None] > 0
    att_h = np.where(xside, g, np.nan)
    att_a = np.where(~xside, 1 - g, np.nan)
    with np.errstate(all="ignore"):
        ah = np.interp(np.arange(n), sub, np.nanmean(att_h, 1))
        aa = np.interp(np.arange(n), sub, np.nanmean(att_a, 1))
    add("area_home", area - 0.5, "anti")
    add("area_near_ball", area_near - 0.5, "anti")
    add("H_att_space", ah, "H:att_space")
    add("A_att_space", aa, "A:att_space")

    # ---- last confident pitch control at the (seen) ball
    ch = raw_seen & (pc_s > CTRL_HI)
    ca = raw_seen & (pc_s < 1 - CTRL_HI)
    code = np.where(ch, 1.0, np.where(ca, -1.0, np.nan))
    idx = np.where(~np.isnan(code), np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    lc = np.where(last >= 0, code[np.maximum(last, 0)], 0.0)
    idxn = np.where(~np.isnan(code), np.arange(n), n)
    nxt = np.minimum.accumulate(idxn[::-1])[::-1]
    nc = np.where(nxt < n, code[np.minimum(nxt, n - 1)], 0.0)
    capc = 30 * fps
    add("pc_last", lc, "anti")
    add("pc_next", nc, "anti")
    add("H_since_pc", np.log1p(since_last(ch, capc)), "H:since_pc")
    add("A_since_pc", np.log1p(since_last(ca, capc)), "A:since_pc")
    add("H_until_pc", np.log1p(until_next(ch, capc)), "H:until_pc")
    add("A_until_pc", np.log1p(until_next(ca, capc)), "A:until_pc")
    # integrated Spearman control over the recent / coming ball path
    add("pcs_path_b", mmean(pc_i - 0.5, -int(1.5 * fps), 0), "anti")
    add("pcs_path_f", mmean(pc_i - 0.5, 0, int(1.5 * fps)), "anti")

    # ---- multi-scale context
    base = {nm: c for nm, c in zip(names, cols)}
    kind_of = dict(zip(names, kinds))
    for key in ("pcs_ball_i", "tti_d1", "pcf_ball", "area_home", "pcf_ring", "pc_last"):
        x = base[key].astype(np.float64)
        for sec in (0.5, 2.0, 5.0):
            hw = int(round(sec * fps))
            add("{}_m{}".format(key, sec), mmean(x, -hw, hw), kind_of[key])
        hw = int(round(2.0 * fps))
        add("{}_pmf".format(key), mmean(x, -hw, -1) - mmean(x, 1, hw), kind_of[key])
    for key in ("pcs_ball_i", "tti_d1"):
        x = mmean(base[key].astype(np.float64), -6, 6)
        for sec in (0.8, 2.0, 4.0):
            k = int(round(sec * fps))
            add("{}_s+{}".format(key, sec), shift(x, k), kind_of[key])
            add("{}_s-{}".format(key, sec), shift(x, -k), kind_of[key])

    F = np.stack(cols, 1).astype(np.float32)
    F[np.isinf(F)] = np.nan
    return F, names, kinds
