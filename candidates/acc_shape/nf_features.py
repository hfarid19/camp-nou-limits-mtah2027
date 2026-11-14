"""ACC + SHAPE input features (29 columns) for the v11 TCN / MS-TCN members (candidates/final_v11).

The code below the imports is the experiment's feature module (round-5 "newfeat" / "acc_ens",
docs/prereg/round5_prereg.txt addenda of 13 Nov 2026), unchanged except that the repo's base feature module is found
relative to this file and that CHUNK (frames per block of the SHAPE pass) is 512 instead of 2048, which lowers peak
memory and leaves every value unchanged. numpy + scipy.ndimage.maximum_filter1d (scipy is in requirements-submission.txt); deterministic.

compute(players, slot_team, ball, sign, pitch, fps, groups) -> F (N, 194 + k), names, kinds
    = candidates/nn_pff/features.py compute(...) (imported unchanged) with the extra columns appended (ACC, then SHAPE).
extra(players, slot_team, ball, pitch, fps, groups, frames=None) -> E (len(frames) or N, k) float32, names, kinds
    ACC is evaluated at every frame (it uses temporal context) and then subset; SHAPE (no temporal context) only at `frames`.

ACC (14 columns)
  Ball: the raw ball with EVERY gap linearly interpolated (ballc / radar_common.interp_ball), then candidates/nn_pointer/
  ballc.py _kinematics (copied verbatim below, sign dropped: only magnitudes are used): central-difference velocity,
  centred 5-frame NaN-aware moving mean, speed; acceleration acc[t] = (speed[t+h] - speed[t-h]) / (2h / fps), h = 0.2 s
  (m/s^2, signed); turn[t] = angle between velocity at t-h and t+h / pi (0 unless both speeds > 0.5 m/s).
  Where the ball is missing (= NaN after the existing 0.5-s short-gap fill, i.e. existing ball_seen / ball_gap flags) both
  are 0. Columns: ball_acc (signed), ball_turn, ball_acc_mx{0.4,1,2} = max |acc| over t +- 0.4 / 1 / 2 s,
  ball_turn_mx{0.4,1,2} = max turn over the same windows (outside the game: 0).
  Players: velocity exactly as the existing speed features (frame difference x fps, a frame with speed > 12 m/s -> NaN =
  the jump filter, centred 5-frame NaN-aware moving mean = fe.mmean2d hw 2); acceleration vector a[t] =
  (v[t+h] - v[t-h]) / (2h / fps), h = 0.2 s (as the ball); magnitude, capped at 15 m/s^2; NaN where either velocity is NaN.
  Per team T (H, A): T_acc_near = |a| of T's player nearest the ball (the existing H_d1 / A_d1 player); T_acc_3m = the
  largest |a| among T's players within 3 m of the ball (0 if none). 0 where the ball is missing, the team has no tracked
  player, or the player's acceleration is NaN. diff_acc_near, diff_acc_3m = H - A.
SHAPE (15 columns; positions as the existing features: players float32, ball = raw ball with gaps <= 0.5 s filled)
  T_hull = area (m^2) of the convex hull of T's tracked players (0 if fewer than 3 or all collinear); diff_hull = H - A;
  hull_overlap = area(hull H intersect hull A) / min(area H, area A), in [0, 1] (0 if either area is 0).
  T_pang = angle (radians, [0, pi]) at the ball between the directions to T's two nearest players; T_ptri = area (m^2) of
  the triangle ball + those two players; 0 if the ball is missing or T has fewer than 2 tracked players; diff_pang,
  diff_ptri = H - A.
  T_lanes = number of T's tracked players within 25 m of T's player nearest the ball (the passer; passer excluded) whose
  segment passer -> teammate has no tracked opponent within 1.5 m (point-to-segment distance); 0 if the ball is missing or
  T has no tracked player. diff_lanes = H - A. lanes_near = the lanes of the team whose nearest player is nearer the ball
  (strict; exact tie -> max of the two), lanes_far = the other team's (tie -> min).
  None of the new columns depends on the attack direction (areas, angles, counts, magnitudes), so no sign is used.
Swap kinds (repo convention): H_x / A_x pairs "H:x" / "A:x", diff_* "anti", everything else "sym".
"""

import importlib.util
import os

import numpy as np
from scipy.ndimage import maximum_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("nf_repo_features", os.path.join(os.path.dirname(HERE), "nn_pff", "features.py"))
fe = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(fe)
swap_index = fe.swap_index
frame_sign = fe.frame_sign

GROUPS = ("acc", "shape")
ACC_CAP = 15.0          # m/s^2, player acceleration magnitude cap
KIN_H_S = 0.2           # half-width of the acceleration difference (= ballc)
WINS = (0.4, 1.0, 2.0)  # max windows (+- s)
NEAR_R = 3.0            # T_acc_3m radius
LANE_R = 25.0           # passing-lane teammate radius
BLOCK_R = 1.5           # opponent within this of the segment blocks the lane
KMIN = 16               # minimum compacted players per team (fixed per game: see _kfix)
CHUNK = 512             # frames per block of the SHAPE pass (memory only: the experiment used 2048; same values)
L_HALF, W_HALF = 52.5, 34.0


# ---------------------------------------------------------------------------------------------------------------------
# verbatim copies (candidates/ball_radar_pff/radar_common.py interp_ball / _mmean_axis0; nn_pointer/ballc.py _kinematics)
# ---------------------------------------------------------------------------------------------------------------------

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


def _kinematics(ball_i, fps):
    """ballc._kinematics without the attack-frame sign: velocity, acceleration, turn angle / pi (full rate)."""
    n = len(ball_i)
    v = np.zeros((n, 2))
    v[1:-1] = (ball_i[2:] - ball_i[:-2]) * fps / 2.0
    v = _mmean_axis0(v, 2)
    sp = np.sqrt((v ** 2).sum(1))
    acc = np.zeros(n)
    h = max(1, int(round(KIN_H_S * fps)))
    acc[h:-h] = (sp[2 * h:] - sp[:-2 * h]) / (2 * h / fps)
    turn = np.zeros(n)
    a, b = v[:-2 * h], v[2 * h:]
    na, nb = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))
    ok = (na > 0.5) & (nb > 0.5)
    cos = np.where(ok, (a * b).sum(1) / np.maximum(na * nb, 1e-9), 1.0)
    turn[h:-h] = np.arccos(np.clip(cos, -1, 1)) / np.pi
    return v, acc, turn


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def _ball_filled(ball_raw, fps):
    """= fe.compute: short gaps (<= 0.5 s) linearly filled; seen = not NaN after the fill."""
    gapf = int(round(fe.GAP_FILL_S * fps))
    ball = np.stack([fe.fill_short_gaps(ball_raw[:, 0].astype(np.float64), gapf),
                     fe.fill_short_gaps(ball_raw[:, 1].astype(np.float64), gapf)], 1)
    return ball, ~np.isnan(ball[:, 0])


def _player_velocity(players, fps):
    """= fe.compute 'player motion' (jump filter 12 m/s, fe.mmean2d hw 2)."""
    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * fps
    spd = np.sqrt((vel ** 2).sum(-1))
    vel[spd > 12.0] = np.nan
    return fe.mmean2d(vel[:, :, 0], 2), fe.mmean2d(vel[:, :, 1], 2)


def _player_acc(vx, vy, fps):
    h = max(1, int(round(KIN_H_S * fps)))
    n = len(vx)
    ax = np.full(vx.shape, np.nan)
    ay = np.full(vy.shape, np.nan)
    if n > 2 * h:
        ax[h:-h] = (vx[2 * h:].astype(np.float64) - vx[:-2 * h]) / (2 * h / fps)
        ay[h:-h] = (vy[2 * h:].astype(np.float64) - vy[:-2 * h]) / (2 * h / fps)
    return np.minimum(np.sqrt(ax ** 2 + ay ** 2), ACC_CAP)       # NaN stays NaN


def _movmax(x, hw):
    """max over t-hw..t+hw of x >= 0 (outside the array: 0)."""
    return maximum_filter1d(np.asarray(x, np.float64), size=2 * hw + 1, mode="constant", cval=0.0)


def _kfix(valid_full):
    return max(KMIN, int(valid_full.sum(1).max()) if len(valid_full) else 0)


def _compact(P, valid, K):
    """(n, S, 2), (n, S) -> the tracked players first (slot order kept), width K: Q (n, K, 2), v (n, K), slot (n, K)."""
    n, S = valid.shape
    if S < K:
        P = np.concatenate([P, np.full((n, K - S, 2), np.nan)], 1)
        valid = np.concatenate([valid, np.zeros((n, K - S), bool)], 1)
        S = K
    order = np.argsort(~valid, axis=1, kind="stable")[:, :K]
    Q = np.take_along_axis(P, order[:, :, None], 1)
    v = np.take_along_axis(valid, order, 1)
    Q = np.where(v[:, :, None], Q, 0.0)
    return Q, v, np.where(v, order, -1)


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _hull(Q, v):
    """Convex hull of the valid points of each frame (Q relative to a common origin).
    Edge i -> j is a hull edge iff every other valid point k is strictly left of it, or on its line within the segment
    (so collinear boundary points do not split an edge); exact duplicate points are dropped first.
    -> area (n,), nxt (n, K) successor of each hull vertex, hv (n, K) vertex has a hull edge."""
    n, K = v.shape
    same = (Q[:, :, None, :] == Q[:, None, :, :]).all(-1) & v[:, :, None] & v[:, None, :]
    earlier = np.triu(np.ones((K, K), bool), 1)                         # [j, k] j < k
    v = v & ~(same & earlier[None]).any(1)
    ex = Q[:, None, :, 0] - Q[:, :, None, 0]                            # [n, i, j] = Q_j - Q_i
    ey = Q[:, None, :, 1] - Q[:, :, None, 1]
    ee = ex ** 2 + ey ** 2
    cr = ex[:, :, :, None] * ey[:, :, None, :] - ey[:, :, :, None] * ex[:, :, None, :]   # [n,i,j,k] cross(e_ij, d_ik)
    dt = ex[:, :, :, None] * ex[:, :, None, :] + ey[:, :, :, None] * ey[:, :, None, :]
    tol = 1e-9 * (ee[:, :, :, None] + ee[:, :, None, :])
    ok = (cr > tol) | ((np.abs(cr) <= tol) & (dt >= -tol) & (dt <= ee[:, :, :, None] + tol)) | ~v[:, None, None, :]
    edge = ok.all(3) & v[:, :, None] & v[:, None, :] & ~np.eye(K, dtype=bool)[None]
    edge &= (v.sum(1) >= 3)[:, None, None]
    cq = _cross(Q[:, :, None, 0], Q[:, :, None, 1], Q[:, None, :, 0], Q[:, None, :, 1])
    area = np.maximum(0.5 * np.where(edge, cq, 0.0).sum((1, 2)), 0.0)
    return area, np.argmax(edge, 2), edge.any(2)


def _clip_len(QA, nA, hA, QB, nB, hB):
    """Green's-theorem contribution of hull A's edges clipped to the inside of hull B (Cyrus-Beck): sum cross(p(t0), p(t1))."""
    p0x, p0y = QA[:, :, 0], QA[:, :, 1]
    p1 = np.take_along_axis(QA, nA[:, :, None], 1)
    dx, dy = p1[:, :, 0] - p0x, p1[:, :, 1] - p0y
    q0x, q0y = QB[:, :, 0], QB[:, :, 1]
    q1 = np.take_along_axis(QB, nB[:, :, None], 1)
    exb, eyb = q1[:, :, 0] - q0x, q1[:, :, 1] - q0y
    num = _cross(exb[:, None, :], eyb[:, None, :], p0x[:, :, None] - q0x[:, None, :], p0y[:, :, None] - q0y[:, None, :])
    den = _cross(exb[:, None, :], eyb[:, None, :], dx[:, :, None], dy[:, :, None])
    tol = 1e-9 * ((exb ** 2 + eyb ** 2)[:, None, :] + (dx ** 2 + dy ** 2)[:, :, None])
    hb = hB[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = -num / den
    enter = np.where(hb & (den > tol), tc, -np.inf).max(2)
    leave = np.where(hb & (den < -tol), tc, np.inf).min(2)
    empty = (hb & (np.abs(den) <= tol) & (num < -tol)).any(2)
    t0, t1 = np.maximum(enter, 0.0), np.minimum(leave, 1.0)
    use = hA & ~empty & (t1 > t0)
    t0, t1 = np.where(use, t0, 0.0), np.where(use, t1, 0.0)
    c = _cross(p0x + t0 * dx, p0y + t0 * dy, p0x + t1 * dx, p0y + t1 * dy)
    return np.where(use, c, 0.0).sum(1)


def _pressure(Q, v, bx, by):
    """Two nearest valid players to the ball: angle at the ball (rad) and triangle area; 0 if fewer than 2."""
    d = np.where(v, np.sqrt((Q[:, :, 0] - bx[:, None]) ** 2 + (Q[:, :, 1] - by[:, None]) ** 2), np.inf)
    o = np.argsort(d, axis=1, kind="stable")[:, :2]
    p = np.take_along_axis(Q, o[:, :, None], 1)
    ok = np.isfinite(np.take_along_axis(d, o, 1)).all(1)
    ux, uy = p[:, 0, 0] - bx, p[:, 0, 1] - by
    wx, wy = p[:, 1, 0] - bx, p[:, 1, 1] - by
    nu, nw = np.sqrt(ux ** 2 + uy ** 2), np.sqrt(wx ** 2 + wy ** 2)
    good = ok & (nu > 1e-6) & (nw > 1e-6)
    cos = np.where(good, (ux * wx + uy * wy) / np.maximum(nu * nw, 1e-12), 1.0)
    ang = np.where(good, np.arccos(np.clip(cos, -1.0, 1.0)), 0.0)
    tri = np.where(ok, 0.5 * np.abs(_cross(ux, uy, wx, wy)), 0.0)
    return ang, tri


def _lanes(Qt, vt, st, jslot, has, Qo, vo):
    """Open lanes from the passer (team slot jslot) to teammates within LANE_R; blocked by an opponent within BLOCK_R."""
    sel = (st == jslot[:, None]) & vt
    px = np.where(sel, Qt[:, :, 0], 0.0).sum(1)
    py = np.where(sel, Qt[:, :, 1], 0.0).sum(1)
    sx, sy = Qt[:, :, 0] - px[:, None], Qt[:, :, 1] - py[:, None]
    ss = sx ** 2 + sy ** 2
    mate = vt & ~sel & (ss <= LANE_R ** 2)
    wx, wy = Qo[:, None, :, 0] - px[:, None, None], Qo[:, None, :, 1] - py[:, None, None]     # [n, m, o]
    t = np.clip((wx * sx[:, :, None] + wy * sy[:, :, None]) / np.maximum(ss, 1e-12)[:, :, None], 0.0, 1.0)
    rx, ry = wx - t * sx[:, :, None], wy - t * sy[:, :, None]
    blocked = ((rx ** 2 + ry ** 2 < BLOCK_R ** 2) & vo[:, None, :]).any(2)
    return np.where(has, (mate & ~blocked).sum(1), 0).astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------------------------

def extra(players, slot_team, ball_raw, pitch, fps, groups, frames=None):
    groups = [g for g in GROUPS if g in groups]
    L, W = pitch
    n = len(ball_raw)
    fr = np.arange(n) if frames is None else np.asarray(frames, np.int64)
    players = players.astype(np.float32)                      # = fe.compute
    home, away = slot_team == 0, slot_team == 1
    ball, seen = _ball_filled(ball_raw, fps)
    px, py = players[:, :, 0], players[:, :, 1]
    dist = np.sqrt((px - ball[:, None, 0]) ** 2 + (py - ball[:, None, 1]) ** 2)     # = fe.compute
    rows = np.arange(n)
    near = {}
    for tag, mask in (("H", home), ("A", away)):
        dd = np.where(np.isnan(dist[:, mask]), np.inf, dist[:, mask])
        j = np.argmin(dd, 1) if mask.any() else np.zeros(n, np.int64)
        near[tag] = (dd, j, np.isfinite(dd[rows, j]) if mask.any() else np.zeros(n, bool))
    cols, names, kinds = [], [], []

    def add(name, values, kind="sym"):
        cols.append(np.asarray(values, np.float32))
        names.append(name)
        kinds.append(kind)

    if "acc" in groups:
        _, bacc, bturn = _kinematics(interp_ball(ball_raw), fps)
        bacc = np.where(seen, bacc, 0.0)
        bturn = np.where(seen, bturn, 0.0)
        add("ball_acc", bacc[fr])
        add("ball_turn", bturn[fr])
        for w in WINS:
            add("ball_acc_mx{:g}".format(w), _movmax(np.abs(bacc), int(round(w * fps)))[fr])
        for w in WINS:
            add("ball_turn_mx{:g}".format(w), _movmax(bturn, int(round(w * fps)))[fr])
        vx, vy = _player_velocity(players, fps)
        am = _player_acc(vx, vy, fps)
        t = {}
        for tag, mask in (("H", home), ("A", away)):
            dd, j, has = near[tag]
            at = am[:, mask]
            an = at[rows, j] if mask.any() else np.zeros(n)
            an = np.where(has & seen & ~np.isnan(an), an, 0.0)
            a3 = np.where((dd < NEAR_R) & ~np.isnan(at), at, -np.inf).max(1) if mask.any() else np.full(n, -np.inf)
            a3 = np.where(seen & np.isfinite(a3), a3, 0.0)
            t[tag] = dict(acc_near=an[fr], acc_3m=a3[fr])
        for key in ("acc_near", "acc_3m"):
            add("H_" + key, t["H"][key], "H:" + key)
            add("A_" + key, t["A"][key], "A:" + key)
        for key in ("acc_near", "acc_3m"):
            add("diff_" + key, t["H"][key] - t["A"][key], "anti")

    if "shape" in groups:
        P = players.astype(np.float64)
        P = np.stack([P[:, :, 0] - L / 2, P[:, :, 1] - W / 2], -1)        # common origin: pitch centre
        valid = ~np.isnan(P[:, :, 0])
        Kt = {tag: _kfix(valid[:, m]) for tag, m in (("H", home), ("A", away))}
        bxa, bya = ball[:, 0] - L / 2, ball[:, 1] - W / 2
        out = {k: np.zeros(len(fr)) for k in ("H_hull", "A_hull", "ovl", "H_pang", "A_pang", "H_ptri", "A_ptri",
                                              "H_lanes", "A_lanes")}
        for c0 in range(0, len(fr), CHUNK):
            f = fr[c0:c0 + CHUNK]
            sl = slice(c0, c0 + len(f))
            sn = seen[f]
            bx, by = np.where(sn, bxa[f], 0.0), np.where(sn, bya[f], 0.0)
            comp, hull = {}, {}
            for tag, m in (("H", home), ("A", away)):
                Q, v, st = _compact(P[f][:, m], valid[f][:, m], Kt[tag])
                comp[tag] = (Q, v, st)
                hull[tag] = _hull(Q, v)
                out[tag + "_hull"][sl] = hull[tag][0]
                ang, tri = _pressure(Q, v, bx, by)
                out[tag + "_pang"][sl] = np.where(sn, ang, 0.0)
                out[tag + "_ptri"][sl] = np.where(sn, tri, 0.0)
            (QH, vH, sH), (QA, vA, sA) = comp["H"], comp["A"]
            aH, nH, hH = hull["H"]
            aA, nA, hA = hull["A"]
            ov = 0.5 * (_clip_len(QH, nH, hH, QA, nA, hA) + _clip_len(QA, nA, hA, QH, nH, hH))
            mn = np.minimum(aH, aA)
            out["ovl"][sl] = np.where(mn > 1e-6, np.clip(ov / np.maximum(mn, 1e-6), 0.0, 1.0), 0.0)
            for tag, (Qt, vt, st), (Qo, vo, _) in (("H", comp["H"], comp["A"]), ("A", comp["A"], comp["H"])):
                _, j, has = near[tag]
                out[tag + "_lanes"][sl] = _lanes(Qt, vt, st, j[f], has[f] & sn, Qo, vo)
        add("H_hull", out["H_hull"], "H:hull")
        add("A_hull", out["A_hull"], "A:hull")
        add("H_pang", out["H_pang"], "H:pang")
        add("A_pang", out["A_pang"], "A:pang")
        add("H_ptri", out["H_ptri"], "H:ptri")
        add("A_ptri", out["A_ptri"], "A:ptri")
        add("H_lanes", out["H_lanes"], "H:lanes")
        add("A_lanes", out["A_lanes"], "A:lanes")
        for key in ("hull", "pang", "ptri", "lanes"):
            add("diff_" + key, out["H_" + key] - out["A_" + key], "anti")
        add("hull_overlap", out["ovl"])
        d1h = np.where(near["H"][2], near["H"][0][rows, near["H"][1]], np.inf)[fr]
        d1a = np.where(near["A"][2], near["A"][0][rows, near["A"][1]], np.inf)[fr]
        lh, la = out["H_lanes"], out["A_lanes"]
        add("lanes_near", np.where(d1h < d1a, lh, np.where(d1a < d1h, la, np.maximum(lh, la))))
        add("lanes_far", np.where(d1h < d1a, la, np.where(d1a < d1h, lh, np.minimum(lh, la))))
    E = np.stack(cols, 1).astype(np.float32) if cols else np.zeros((len(fr), 0), np.float32)
    return E, names, kinds


def compute(players, slot_team, ball_raw, sign, pitch, fps, groups=GROUPS):
    F, names, kinds = fe.compute(players, slot_team, ball_raw, sign, pitch, fps)
    E, en, ek = extra(players, slot_team, ball_raw, pitch, fps, groups)
    return np.concatenate([F, E], 1), names + en, kinds + ek
