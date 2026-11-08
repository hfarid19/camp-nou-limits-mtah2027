"""ball_radar copy of learned_dfl_evidence/features.py with the ball-gap positions taken from the
Ball Radar imputed ball (compute(..., ball_imp=...)) and radar evidence columns appended.

Original docstring: shared, identity-free, team-relative per-frame features.

compute(players, slot_team, ball, sign, pitch, fps) -> (F float32 (N, K), names, kinds)

players (N, S, 2) metres NaN-missing, slot_team (S,) 0 home 1 away, ball (N, 2) NaN-missing,
sign (N,) +1 where home attacks towards x = length. Nothing depends on slot identity: every
team feature is an order statistic, count or mean over that team's slots.

kinds[k] says how column k changes when home and away are swapped (team-swap augmentation):
    "sym"  unchanged          "anti"  negated
    "H:<base>" / "A:<base>"   swapped with its partner column
swap_index(names, kinds) returns (perm, signs) with F_swapped = F[:, perm] * signs.
"""

import numpy as np

CTRL_IN, CTRL_OUT, CTRL_PERSIST = 1.0, 1.5, 3
GAP_FILL_S = 0.5


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------

def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def fill_short_gaps(values, max_gap):
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
    """NaN-aware mean of values[t+lo_off : t+hi_off+1] (offsets may be negative), per t."""
    values = np.asarray(values, np.float64)
    v = np.where(np.isnan(values), 0.0, values)
    c = (~np.isnan(values)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v, axis=0)))
    cc = np.concatenate(([0.0], np.cumsum(c, axis=0)))
    n = len(values)
    i = np.arange(n)
    lo = np.clip(i + lo_off, 0, n)
    hi = np.clip(i + hi_off + 1, 0, n)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[hi] - cv[lo]) / np.maximum(cnt, 1), np.nan)


def mmean2d(values, hw):
    """Centred NaN-aware moving mean along axis 0 of a 2-D array."""
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
    """y[t] = x[t + k] with NaN padding."""
    y = np.full_like(x, np.nan)
    if k > 0:
        y[:-k] = x[k:]
    elif k < 0:
        y[-k:] = x[:k]
    else:
        y[:] = x
    return y


def since_last(mask, cap):
    """Frames since the last True (inclusive -> 0), capped."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(n) - last, cap).astype(np.float64)
    return np.minimum(out, cap)


def until_next(mask, cap):
    return since_last(mask[::-1], cap)[::-1]


def kth_smallest(d, k):
    """k-th smallest (0-based) along axis 1 ignoring NaN (NaN if fewer)."""
    s = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
    if s.shape[1] <= k:
        return np.full(len(d), np.nan)
    v = s[:, k]
    return np.where(np.isinf(v), np.nan, v)


# ------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------

def compute(players, slot_team, ball_raw, sign, pitch, fps, ball_imp=None, radar=None):
    L, W = pitch
    n = len(ball_raw)
    players = players.astype(np.float32)
    home, away = slot_team == 0, slot_team == 1
    cols, names, kinds = [], [], []

    def add(name, values, kind="sym"):
        cols.append(np.asarray(values, np.float32))
        names.append(name)
        kinds.append(kind)

    # ---- ball
    raw_seen = ~np.isnan(ball_raw[:, 0])
    gapf = int(round(GAP_FILL_S * fps))
    ball = np.stack([fill_short_gaps(ball_raw[:, 0].astype(np.float64), gapf),
                     fill_short_gaps(ball_raw[:, 1].astype(np.float64), gapf)], 1)
    if ball_imp is not None:                      # Ball Radar: fill the remaining gaps
        miss = np.isnan(ball[:, 0]) & ~np.isnan(ball_imp[:, 0])
        ball[miss] = ball_imp[miss]
    seen = ~np.isnan(ball[:, 0])
    cap = 60 * fps
    since = since_last(raw_seen, cap)
    until = until_next(raw_seen, cap)
    gap_len = np.where(raw_seen, 0.0, np.minimum(since + until - 1, cap))
    add("ball_seen", raw_seen.astype(float))
    add("ball_since", np.log1p(since))
    add("ball_until", np.log1p(until))
    add("ball_gap", np.log1p(gap_len))
    step = np.full(n, np.nan)
    step[1:] = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(1)) * fps
    step[step > 45] = np.nan                         # detector jumps
    bspeed = mmean(step, -2, 2)
    add("ball_speed", bspeed)
    k = int(round(0.5 * fps))
    disp_b = np.sqrt(((ball - np.vstack([np.full((k, 2), np.nan), ball[:-k]])) ** 2).sum(1))
    disp_f = np.sqrt(((np.vstack([ball[k:], np.full((k, 2), np.nan)]) - ball) ** 2).sum(1))
    add("ball_disp_b", disp_b)
    add("ball_disp_f", disp_f)
    bx, by = ball[:, 0], ball[:, 1]
    ox = np.maximum(np.maximum(-bx, bx - L), 0.0)
    oy = np.maximum(np.maximum(-by, by - W), 0.0)
    out_dist = np.sqrt(ox ** 2 + oy ** 2)
    add("ball_out", out_dist)
    add("ball_line", np.minimum(np.minimum(bx, L - bx), np.minimum(by, W - by)))
    add("ball_xdir", (bx - L / 2) * sign, "anti")       # + = towards the goal home attacks
    add("ball_yabs", np.abs(by - W / 2))

    # ---- player motion
    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * fps
    spd = np.sqrt((vel ** 2).sum(-1))
    bad = spd > 12.0
    vel[bad] = np.nan
    vx = mmean2d(vel[:, :, 0], 2)
    vy = mmean2d(vel[:, :, 1], 2)
    spd = np.sqrt(vx ** 2 + vy ** 2)

    tracked = (~np.isnan(players[:, :, 0])).sum(1).astype(float)
    add("n_tracked", tracked)
    add("mean_speed", np.nanmean(spd, 1))
    with np.errstate(all="ignore"):
        add("p90_speed", _nanq(spd, 0.9))
    add("stationary", np.nanmean(np.where(np.isnan(spd), np.nan, (spd < 0.7).astype(float)), 1))
    px, py = players[:, :, 0], players[:, :, 1]
    beyond = ((py < -0.5) | (py > W + 0.5) | (px < -0.5) | (px > L + 0.5)).sum(1)
    add("beyond_line", beyond.astype(float))
    corner = np.zeros(n)
    for cx in (0.0, L):
        for cy in (0.0, W):
            corner += (np.sqrt((px - cx) ** 2 + (py - cy) ** 2) < 3.0).sum(1)
    add("near_corner", corner)
    # players near each goal box; box at x = L/2 - sign*... : "home defends" goal at x where
    # home attacks the other end: home defends x = 0 when sign = +1
    in_box_lo = ((px < 16.5) & (np.abs(py - W / 2) < 20.2)).sum(1).astype(float)
    in_box_hi = ((px > L - 16.5) & (np.abs(py - W / 2) < 20.2)).sum(1).astype(float)
    add("box_homedef", np.where(sign > 0, in_box_lo, in_box_hi), "H:box_def")
    add("box_awaydef", np.where(sign > 0, in_box_hi, in_box_lo), "A:box_def")

    # ---- per-team blocks
    dist = np.sqrt((px - ball[:, None, 0]) ** 2 + (py - ball[:, None, 1]) ** 2)
    team = {}
    for tag, mask, s in (("H", home, 1.0), ("A", away, -1.0)):
        d = dist[:, mask]
        d1, d2, d3 = kth_smallest(d, 0), kth_smallest(d, 1), kth_smallest(d, 2)
        dd = np.where(np.isnan(d), np.inf, d)
        c2, c5, c10 = [(dd < r).sum(1).astype(float) for r in (2.0, 5.0, 10.0)]
        c2[~seen], c5[~seen], c10[~seen] = np.nan, np.nan, np.nan
        j = np.argmin(dd, 1)
        rows = np.arange(n)
        pnx, pny = px[:, mask][rows, j], py[:, mask][rows, j]
        nvx, nvy = vx[:, mask][rows, j], vy[:, mask][rows, j]
        nspd = np.sqrt(nvx ** 2 + nvy ** 2)
        rx, ry = bx - pnx, by - pny
        rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-3
        closing = (nvx * rx + nvy * ry) / rn
        nspd[~seen], closing[~seen] = np.nan, np.nan
        xd = (px[:, mask] - L / 2) * (sign[:, None] * s)     # + = towards the goal it attacks
        yy = py[:, mask]
        with np.errstate(all="ignore"):
            cx = np.nanmean(xd, 1)
            spread = np.nanstd(xd, 1) + np.nanstd(yy, 1)
            length = np.nanmax(xd, 1) - np.nanmin(xd, 1)
            width = np.nanmax(yy, 1) - np.nanmin(yy, 1)
            fwd = np.nanmean(np.clip(vx[:, mask] * (sign[:, None] * s), -10, 10), 1)
        xs = np.sort(np.where(np.isnan(xd), np.inf, xd), 1)
        second = np.where(np.isinf(xs[:, 1]), np.nan, xs[:, 1])
        final3 = (xd > L / 6).sum(1).astype(float)
        ntr = (~np.isnan(xd)).sum(1).astype(float)
        team[tag] = dict(d1=d1, d2=d2, d3=d3, c2=c2, c5=c5, c10=c10, nspd=nspd, close=closing,
                         cx=cx, spread=spread, length=length, width=width, fwd=fwd,
                         defline=second, final3=final3, ntr=ntr)
    for key in team["H"]:
        add("H_" + key, team["H"][key], "H:" + key)
        add("A_" + key, team["A"][key], "A:" + key)
    for key in ("d1", "d2", "c2", "c5", "cx", "spread", "length", "fwd", "defline", "final3",
                "close"):
        add("diff_" + key, team["H"][key] - team["A"][key], "anti")
    add("dmin", np.fmin(team["H"]["d1"], team["A"]["d1"]))

    # ---- last confident control
    dh = np.nan_to_num(team["H"]["d1"], nan=99.0)
    da = np.nan_to_num(team["A"]["d1"], nan=99.0)
    ch = raw_seen & (dh < CTRL_IN) & (da > CTRL_OUT)
    ca = raw_seen & (da < CTRL_IN) & (dh > CTRL_OUT)
    def persist(m):
        out = m.copy()
        for k in range(1, CTRL_PERSIST):
            out &= np.concatenate((np.zeros(k, bool), m[:-k]))
        return out
    ch, ca = persist(ch), persist(ca)
    code = np.where(ch, 1.0, np.where(ca, -1.0, np.nan))
    idx = np.where(~np.isnan(code), np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    lc = np.where(last >= 0, code[np.maximum(last, 0)], 0.0)
    idxn = np.where(~np.isnan(code), np.arange(n), n)
    nxt = np.minimum.accumulate(idxn[::-1])[::-1]
    nc = np.where(nxt < n, code[np.minimum(nxt, n - 1)], 0.0)
    capc = 30 * fps
    add("lastctrl", lc, "anti")
    add("nextctrl", nc, "anti")
    add("H_since_ctrl", np.log1p(since_last(ch, capc)), "H:since_ctrl")
    add("A_since_ctrl", np.log1p(since_last(ca, capc)), "A:since_ctrl")
    add("H_until_ctrl", np.log1p(until_next(ch, capc)), "H:until_ctrl")
    add("A_until_ctrl", np.log1p(until_next(ca, capc)), "A:until_ctrl")
    add("any_since_ctrl", np.log1p(since_last(ch | ca, capc)))

    # ---- multi-scale context
    base = {nm: c for nm, c in zip(names, cols)}
    kind_of = dict(zip(names, kinds))
    ctx_keys = ["ball_seen", "ball_speed", "ball_out", "mean_speed", "stationary", "n_tracked",
                "diff_d1", "diff_c5", "diff_cx", "diff_fwd", "diff_spread", "ball_xdir",
                "lastctrl", "dmin", "p90_speed"]
    for key in ctx_keys:
        x = base[key].astype(np.float64)
        kd = kind_of[key]
        for sec in (0.5, 2.0, 5.0):
            hw = int(round(sec * fps))
            add("{}_m{}".format(key, sec), mmean(x, -hw, hw), kd)
        hw = int(round(2.0 * fps))
        add("{}_pmf".format(key), mmean(x, -hw, -1) - mmean(x, 1, hw), kd)
    shift_keys = ["ball_seen", "mean_speed", "ball_out", "diff_d1", "lastctrl", "stationary",
                  "n_tracked", "dmin"]
    for key in shift_keys:
        x = mmean(base[key].astype(np.float64), -6, 6)
        kd = kind_of[key]
        for sec in (0.8, 2.0, 4.0, 20.0):
            k = int(round(sec * fps))
            add("{}_s+{}".format(key, sec), shift(x, k), kd)
            add("{}_s-{}".format(key, sec), shift(x, -k), kd)

    if radar is not None:                         # Ball Radar evidence columns
        eps = 1e-4
        rt = np.clip(radar["team"], eps, 1.0)
        rl = np.log(rt[:, 0]) - np.log(rt[:, 1])
        rd = np.log(rt[:, 2]) - np.log(np.clip(1 - rt[:, 2], eps, 1.0))
        add("r_team", rl, "anti")
        add("r_dead", rd)
        add("r_gdead", radar["g_dead"])
        add("r_poss_h", radar["poss_h"], "H:r_poss")
        add("r_poss_a", radar["poss_a"], "A:r_poss")
        add("r_imputed", (~raw_seen).astype(float))
        for sec in (0.5, 2.0):
            hw = int(round(sec * fps))
            add("r_team_m{}".format(sec), mmean(rl, -hw, hw), "anti")
            add("r_dead_m{}".format(sec), mmean(rd, -hw, hw))
        hw = int(round(1.0 * fps))
        add("r_team_pmf", mmean(rl, -hw, -1) - mmean(rl, 1, hw), "anti")
    F = np.stack(cols, 1).astype(np.float32)
    F[np.isinf(F)] = np.nan
    return F, names, kinds


def _nanq(a, q):
    s = np.sort(np.where(np.isnan(a), np.inf, a), 1)
    cnt = (~np.isnan(a)).sum(1)
    k = np.clip(np.floor(q * (cnt - 1)).astype(int), 0, a.shape[1] - 1)
    v = s[np.arange(len(a)), k]
    return np.where(cnt > 0, v, np.nan)


def swap_index(names, kinds):
    where = {kd: i for i, kd in enumerate(kinds)}
    perm = np.arange(len(names))
    signs = np.ones(len(names), np.float32)
    for i, kd in enumerate(kinds):
        if kd == "anti":
            signs[i] = -1.0
        elif kd[:2] in ("H:", "A:"):
            perm[i] = where[("A:" if kd[:2] == "H:" else "H:") + kd[2:]]
    return perm, signs


def frame_sign(n, sign_half1, blocks):
    sign = np.full(n, float(sign_half1))
    for half, first, last in blocks:
        if half == 2:
            sign[first:] = -float(sign_half1)
    return sign
