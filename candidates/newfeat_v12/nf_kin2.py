"""v12 member copy (candidates/newfeat_v12): the newfeat experiment's module (arch/r5_arch/newfeat/code), unchanged except
that nf_features is the sibling file loaded by path. newfeat KIN2 group (EXPERIMENT ONLY, not in the repo): extended ball kinematics, tested ON TOP of ACC (prereg addendum
2026-11-13 KIN2). Separate module; reuses nf_features helpers (nf_features.py / nf_rel.py untouched).

kin2(players, slot_team, ball_raw, pitch, fps, frames=None) -> E (len(frames) or N, 26) float32, names, kinds
All columns computed at every frame, then subset. Conventions = ACC: players float32; ball kinematics from the raw ball with
EVERY gap linearly interpolated (radar_common.interp_ball) and ballc._kinematics' velocity v (central difference, centred
5-frame moving mean) and smoothed speed sp; "seen" = ball not NaN after the existing 0.5-s short-gap fill (existing
ball_seen / ball_gap flags); EVERY KIN2 column is 0 where the ball is missing at that step (= the ACC mask). No column
depends on the attack direction (magnitudes, angles, ratios, counts, team signs). H_x / A_x pairs swap; touch_last_team,
touch_next_team anti; the rest sym. h = 0.2 s (5 frames).

(1) jerk (2): vector acceleration a = (v[t+h] - v[t-h]) / 2h, vector jerk j = (a[t+h] - a[t-h]) / 2h (same central
    difference = the smoothing ACC used); ball_jerk = log1p(|j|) (|j| in m/s^3), ball_jerk_mx0.4 = max over t +- 0.4 s.
(2) multi-scale (6): ball_turn_s{0.2,1} = angle between v at t - s/2 and t + s/2, / pi (0 unless both speeds > 0.5 m/s);
    span 0.4 s SKIPPED (= ACC ball_turn). ball_spdratio_s{0.2,0.4,1} = log((sp[t+s/2] + 0.5) / (sp[t-s/2] + 0.5)).
    ball_curv = |v x a| / max(|v|, 1)^3 (1/m), clipped to [0, 2].
(3) touches (9): detector fixed in advance = REL's: s = max(turn / 0.15, |acc| / 12) (ACC's turn / acc), event = s >= 1 and
    s(t) = max over t +- h; touch = event with a tracked player within 1.5 m of the ball at some frame of t +- h; toucher's
    team = the team with the strictly smaller minimum ball distance over t +- h (that team's distance < 1.5 m), tie -> no
    touch; touch distance = that minimum distance.
    touch_since / touch_until = log1p(frames since the last / until the next touch), cap 60 s (inclusive: 0 at a touch);
    touch_last_team / touch_next_team = +1 home, -1 away, 0 none; touch_last_dist / touch_next_dist = toucher distance (m)
    at that touch (0 none); H_touch_b2 / A_touch_b2 = touches of H / A in [t - 2 s, t]; touch_rate_b2 = H + A.
(4) ball-carrier coupling (9): per team T, j = T's player nearest the ball (the H_d1 / A_d1 player), u = j's existing
    smoothed velocity: T_bp_dv = |v - u| (m/s, capped at 40: interpolated-ball jumps); T_bp_align = cos(v, u) (0 unless |v| > 0.5 and |u| > 0.5); diff_bp_dv = H - A.
    ball_fast_ratio = log((|v| + 0.5) / (max speed of the players within 10 m of the ball + 0.5)) (no player within 10 m ->
    denominator 0.5); ball_fast_ratio5 the same with 5 m. opp_close = closing speed (REL's relclose definition, +-40 m/s
    cap) of the nearest player of the team that does NOT own the overall nearest player (exact tie: mean of both teams').
    near_close = the same for the owning team's nearest player.
Skipped (already in base or ACC): ball speed (base ball_speed), tangential acceleration and its maxima (ACC ball_acc,
ball_acc_mx*), turn over 0.4 s and its maxima (ACC ball_turn, ball_turn_mx*), player-velocity closing speed (base H_close /
A_close / diff_close), displacement over 0.5 s (base ball_disp_b / f).
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

import importlib.util  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_NFN = "mtah_newfeat_v12_nf_features"
if _NFN not in sys.modules:                     # the sibling nf_features.py, loaded once per process under a private name
    _s = importlib.util.spec_from_file_location(_NFN, os.path.join(_HERE, "nf_features.py"))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_NFN] = _m
    _s.loader.exec_module(_m)
nf = sys.modules[_NFN]

fe = nf.fe
H_S = 0.2
TURN_THR, ACC_THR, TOUCH_R = 0.15, 12.0, 1.5
RC_CAP = 40.0
CAP_S = 60


def _turn_span(v, sp, k):
    n = len(v)
    out = np.zeros(n)
    if n > 2 * k:
        a, b = v[:-2 * k], v[2 * k:]
        na, nb = sp[:-2 * k], sp[2 * k:]
        ok = (na > 0.5) & (nb > 0.5)
        cos = np.where(ok, (a * b).sum(1) / np.maximum(na * nb, 1e-9), 1.0)
        out[k:-k] = np.arccos(np.clip(cos, -1, 1)) / np.pi
    return out


def _ratio_span(sp, k):
    n = len(sp)
    out = np.zeros(n)
    if n > 2 * k:
        out[k:-k] = np.log((sp[2 * k:] + 0.5) / (sp[:-2 * k] + 0.5))
    return out


def _cdiff(x, h, fps):
    out = np.zeros(x.shape)
    if len(x) > 2 * h:
        out[h:-h] = (x[2 * h:] - x[:-2 * h]) / (2 * h / fps)
    return out


def _last_value(mask, vals):
    """value of vals at the last True of mask at or before t (0 if none)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, vals[np.maximum(last, 0)], 0.0)


def _next_value(mask, vals):
    return _last_value(mask[::-1], vals[::-1])[::-1]


def kin2(players, slot_team, ball_raw, pitch, fps, frames=None):
    n = len(ball_raw)
    fr = np.arange(n) if frames is None else np.asarray(frames, np.int64)
    players = players.astype(np.float32)
    home, away = slot_team == 0, slot_team == 1
    ball, seen = nf._ball_filled(ball_raw, fps)
    px, py = players[:, :, 0], players[:, :, 1]
    dist = np.sqrt((px - ball[:, None, 0]) ** 2 + (py - ball[:, None, 1]) ** 2)     # = fe.compute
    rows = np.arange(n)
    h = max(1, int(round(H_S * fps)))
    vx, vy = nf._player_velocity(players, fps)
    pspd = np.sqrt(vx.astype(np.float64) ** 2 + vy.astype(np.float64) ** 2)
    v, acc, turn = nf._kinematics(nf.interp_ball(ball_raw), fps)
    sp = np.sqrt((v ** 2).sum(1))
    cols, names, kinds = [], [], []

    def add(name, values, kind="sym"):
        x = np.where(seen, np.nan_to_num(np.asarray(values, np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        cols.append(x[fr].astype(np.float32))
        names.append(name)
        kinds.append(kind)

    # (1) jerk
    av = _cdiff(v, h, fps)
    jv = _cdiff(av, h, fps)
    jerk = np.where(seen, np.log1p(np.sqrt((jv ** 2).sum(1))), 0.0)
    add("ball_jerk", jerk)
    add("ball_jerk_mx0.4", maximum_filter1d(jerk, size=2 * int(round(0.4 * fps)) + 1, mode="constant", cval=0.0))
    # (2) multi-scale
    for s in (0.2, 1.0):
        add("ball_turn_s{:g}".format(s), _turn_span(v, sp, max(1, int(round(s * fps / 2)))))
    for s in (0.2, 0.4, 1.0):
        add("ball_spdratio_s{:g}".format(s), _ratio_span(sp, max(1, int(round(s * fps / 2)))))
    cr = np.abs(v[:, 0] * av[:, 1] - v[:, 1] * av[:, 0])
    add("ball_curv", np.clip(cr / np.maximum(sp, 1.0) ** 3, 0.0, 2.0))
    # (3) touches
    near = {}
    for tag, mask in (("H", home), ("A", away)):
        if mask.any():
            dd = np.where(np.isnan(dist[:, mask]), np.inf, dist[:, mask])
            j = np.argmin(dd, 1)
            d1 = dd[rows, j]
            has = np.isfinite(d1)
            slot = np.arange(players.shape[1])[mask][j]
        else:
            d1, has, slot = np.full(n, np.inf), np.zeros(n, bool), np.zeros(n, np.int64)
        near[tag] = (d1, has, slot)
    accm = np.where(seen, acc, 0.0)
    turnm = np.where(seen, turn, 0.0)
    s = np.maximum(turnm / TURN_THR, np.abs(accm) / ACC_THR)
    ev = (s >= 1.0) & (s >= maximum_filter1d(s, size=2 * h + 1, mode="constant", cval=0.0))
    wH = minimum_filter1d(np.where(np.isfinite(near["H"][0]), near["H"][0], 1e9), size=2 * h + 1, mode="nearest")
    wA = minimum_filter1d(np.where(np.isfinite(near["A"][0]), near["A"][0], 1e9), size=2 * h + 1, mode="nearest")
    tH = ev & (wH < TOUCH_R) & (wH < wA)
    tA = ev & (wA < TOUCH_R) & (wA < wH)
    tt = tH | tA
    team = np.where(tH, 1.0, np.where(tA, -1.0, 0.0))
    tdist = np.where(tH, wH, np.where(tA, wA, 0.0))
    cap = CAP_S * fps
    add("touch_since", np.log1p(fe.since_last(tt, cap)))
    add("touch_until", np.log1p(fe.until_next(tt, cap)))
    add("touch_last_team", _last_value(tt, team), "anti")
    add("touch_next_team", _next_value(tt, team), "anti")
    add("touch_last_dist", _last_value(tt, tdist))
    add("touch_next_dist", _next_value(tt, tdist))
    k2 = int(round(2.0 * fps))
    cH = np.concatenate(([0.0], np.cumsum(tH.astype(np.float64))))
    cA = np.concatenate(([0.0], np.cumsum(tA.astype(np.float64))))
    lo = np.clip(rows - k2, 0, n)
    bH, bA = cH[rows + 1] - cH[lo], cA[rows + 1] - cA[lo]
    add("H_touch_b2", bH, "H:touch_b2")
    add("A_touch_b2", bA, "A:touch_b2")
    add("touch_rate_b2", bH + bA)
    # (4) ball-carrier coupling
    t4 = {}
    for tag in ("H", "A"):
        d1, has, slot = near[tag]
        ux, uy = vx[rows, slot].astype(np.float64), vy[rows, slot].astype(np.float64)
        okv = has & np.isfinite(ux) & np.isfinite(uy)
        dv = np.sqrt((v[:, 0] - ux) ** 2 + (v[:, 1] - uy) ** 2)
        nu = np.sqrt(ux ** 2 + uy ** 2)
        al = np.where(okv & (sp > 0.5) & (nu > 0.5), (v[:, 0] * ux + v[:, 1] * uy) / np.maximum(sp * nu, 1e-9), 0.0)
        t4[tag] = (np.where(okv, np.minimum(dv, RC_CAP), 0.0), np.where(okv, al, 0.0))
    add("H_bp_dv", t4["H"][0], "H:bp_dv")
    add("A_bp_dv", t4["A"][0], "A:bp_dv")
    add("H_bp_align", t4["H"][1], "H:bp_align")
    add("A_bp_align", t4["A"][1], "A:bp_align")
    add("diff_bp_dv", t4["H"][0] - t4["A"][0], "anti")
    for r, nm in ((10.0, "ball_fast_ratio"), (5.0, "ball_fast_ratio5")):
        within = (dist < r) & np.isfinite(pspd)
        mx = np.where(within, pspd, 0.0).max(1) if players.shape[1] else np.zeros(n)
        add(nm, np.log((sp + 0.5) / (mx + 0.5)))
    rc = {}
    for tag in ("H", "A"):
        d1, has, slot = near[tag]
        dm = nf_rel_shift(dist, rows - h, slot)
        dp = nf_rel_shift(dist, rows + h, slot)
        c = -(dp - dm) / (2 * h / fps)
        rc[tag] = np.clip(np.where(has & np.isfinite(c), c, 0.0), -RC_CAP, RC_CAP)
    dH, dA = near["H"][0], near["A"][0]
    opp = np.where(dH < dA, rc["A"], np.where(dA < dH, rc["H"], 0.5 * (rc["H"] + rc["A"])))
    own = np.where(dH < dA, rc["H"], np.where(dA < dH, rc["A"], 0.5 * (rc["H"] + rc["A"])))
    add("opp_close", opp)
    add("near_close", own)
    E = np.stack(cols, 1).astype(np.float32)
    return E, names, kinds


def nf_rel_shift(arr2, rr, cc):
    m = arr2.shape[0]
    ok = (rr >= 0) & (rr < m)
    x = arr2[np.clip(rr, 0, m - 1), cc]
    return np.where(ok, x, np.nan)
