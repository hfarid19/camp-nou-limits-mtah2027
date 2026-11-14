"""v12 member copy (candidates/newfeat_v12): the newfeat experiment's module (arch/r5_arch/newfeat/code), unchanged except
that nf_features is the sibling file loaded by path. newfeat REL group (EXPERIMENT ONLY, not in the repo): relational, touch-like per-step features (prereg addendum
2026-11-13 REL). Separate module so nf_features.py (ACC / SHAPE, used by running scorers) is untouched; reuses its helpers.

rel(players, slot_team, ball_raw, pitch, fps, frames=None) -> E (len(frames) or N, 19) float32, names, kinds
All columns are computed at every frame (they use temporal context) and then subset to `frames`.
Conventions = nf_features: players float32; "ball" = raw ball with gaps <= 0.5 s filled (fe), missing = existing
ball_seen / ball_gap flags; every REL column is 0 where the ball is missing at that step; no column depends on the attack
direction; H_x / A_x pairs swap, diff_* and touch are anti, the rest sym. h = 0.2 s (5 frames at 25 fps).

  T_relclose      closing speed (m/s, + = approaching) between the ball and T's player nearest the ball at t (the H_d1 /
                  A_d1 player j(t)): -(dist_j(t+h) - dist_j(t-h)) / 2h, the same slot j(t) at both ends, clipped to +-40 m/s; 0 if undefined.
                  diff_relclose = H - A.
  ball event      ballc kinematics as ACC (acc m/s^2, turn = angle / pi): s(t) = max(turn / 0.15, |acc| / 12);
                  event at t iff s(t) >= 1 and s(t) = max of s over t +- h (a peak).
  touch           +1 if a ball event at t has a home player within 1.5 m of the ball at some frame of t +- h and that
                  team's minimum distance over t +- h is strictly smaller than the away team's; -1 the same for away; else 0.
  T_touch_b1/b3   number of T's touches in [t - 1 s, t - 1 frame] / [t - 3 s, ...];  T_touch_f1/f3 in [t + 1 frame, t + 1 s] /
                  [..., t + 3 s] (outside the game: none).
  nn_chg_b1/b3    number of frames in (t - 1 s, t] / (t - 3 s, t] where the overall nearest tracked player to the ball (slot)
                  differs from the previous frame's (both frames with ball and a tracked player);
  nnteam_chg_b1/b3 the same for the nearest player's team.
  opp_dist        distance (m) between H's and A's players nearest the ball; opp_relspeed = |v_H - v_A| (m/s) of those
                  two players (velocities = the existing smoothed velocities); 0 if either is missing or undefined.
  ball_since_turn log1p(frames since the last frame with turn >= 0.15), capped at 60 s.
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
TURN_THR = 0.15
ACC_THR = 12.0
TOUCH_R = 1.5
RC_CAP = 40.0          # m/s, |relclose| cap (ball detector jumps)
WINS = (1.0, 3.0)


def _sum_window(x, lo, hi):
    """sum of x[t+lo .. t+hi] (inclusive offsets; clipped to the array)."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    n = len(x)
    i = np.arange(n)
    a = np.clip(i + lo, 0, n)
    b = np.clip(i + hi + 1, 0, n)
    return np.where(b > a, c[b] - c[a], 0.0)


def _shift_take(arr2, rows, cols):
    """arr2[rows, cols] with rows clipped; NaN where rows out of range."""
    n = arr2.shape[0]
    ok = (rows >= 0) & (rows < n)
    v = arr2[np.clip(rows, 0, n - 1), cols]
    return np.where(ok, v, np.nan)


def rel(players, slot_team, ball_raw, pitch, fps, frames=None):
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
    slots_all = np.arange(players.shape[1])
    cols, names, kinds = [], [], []

    def add(name, values, kind="sym"):
        v = np.where(seen, np.nan_to_num(np.asarray(values, np.float64), nan=0.0), 0.0)
        cols.append(v[fr].astype(np.float32))
        names.append(name)
        kinds.append(kind)

    near = {}
    for tag, mask in (("H", home), ("A", away)):
        if mask.any():
            dd = np.where(np.isnan(dist[:, mask]), np.inf, dist[:, mask])
            j = np.argmin(dd, 1)
            d1 = dd[rows, j]
            has = np.isfinite(d1)
            slot = slots_all[mask][j]
        else:
            d1, has, slot = np.full(n, np.inf), np.zeros(n, bool), np.zeros(n, np.int64)
        near[tag] = (d1, has, slot)
    # ---- closing speed ball <-> T's nearest player (same slot at t-h and t+h)
    rc = {}
    for tag in ("H", "A"):
        d1, has, slot = near[tag]
        dm = _shift_take(dist, rows - h, slot)
        dp = _shift_take(dist, rows + h, slot)
        c = -(dp - dm) / (2 * h / fps)
        rc[tag] = np.clip(np.where(has & np.isfinite(c), c, 0.0), -RC_CAP, RC_CAP)
    add("H_relclose", rc["H"], "H:relclose")
    add("A_relclose", rc["A"], "A:relclose")
    add("diff_relclose", rc["H"] - rc["A"], "anti")
    # ---- ball events and touches
    _, acc, turn = nf._kinematics(nf.interp_ball(ball_raw), fps)
    acc = np.where(seen, acc, 0.0)
    turn = np.where(seen, turn, 0.0)
    s = np.maximum(turn / TURN_THR, np.abs(acc) / ACC_THR)
    ev = (s >= 1.0) & (s >= maximum_filter1d(s, size=2 * h + 1, mode="constant", cval=0.0))
    wH = minimum_filter1d(np.where(np.isfinite(near["H"][0]), near["H"][0], 1e9), size=2 * h + 1, mode="nearest")
    wA = minimum_filter1d(np.where(np.isfinite(near["A"][0]), near["A"][0], 1e9), size=2 * h + 1, mode="nearest")
    tH = (ev & (wH < TOUCH_R) & (wH < wA)).astype(np.float64)
    tA = (ev & (wA < TOUCH_R) & (wA < wH)).astype(np.float64)
    add("touch", tH - tA, "anti")
    for w in WINS:
        k = int(round(w * fps))
        tag = "{:g}".format(w)
        add("H_touch_b" + tag, _sum_window(tH, -k, -1), "H:touch_b" + tag)
        add("A_touch_b" + tag, _sum_window(tA, -k, -1), "A:touch_b" + tag)
        add("H_touch_f" + tag, _sum_window(tH, 1, k), "H:touch_f" + tag)
        add("A_touch_f" + tag, _sum_window(tA, 1, k), "A:touch_f" + tag)
    # ---- changes of the overall nearest player / its team
    dall = np.where(np.isnan(dist), np.inf, dist)
    jn = np.argmin(dall, 1)
    ok = seen & np.isfinite(dall[rows, jn])
    tn = slot_team[jn]
    both = ok.copy()
    both[1:] &= ok[:-1]
    both[0] = False
    chg = np.zeros(n)
    tchg = np.zeros(n)
    chg[1:] = (both[1:] & (jn[1:] != jn[:-1]))
    tchg[1:] = (both[1:] & (tn[1:] != tn[:-1]))
    for w in WINS:
        k = int(round(w * fps))
        add("nn_chg_b{:g}".format(w), _sum_window(chg, -(k - 1), 0))
    for w in WINS:
        k = int(round(w * fps))
        add("nnteam_chg_b{:g}".format(w), _sum_window(tchg, -(k - 1), 0))
    # ---- the two nearest opposing players
    sH, sA = near["H"][2], near["A"][2]
    okp = near["H"][1] & near["A"][1]
    od = np.sqrt((px[rows, sH] - px[rows, sA]) ** 2 + (py[rows, sH] - py[rows, sA]) ** 2).astype(np.float64)
    rs = np.sqrt((vx[rows, sH] - vx[rows, sA]) ** 2 + (vy[rows, sH] - vy[rows, sA]) ** 2).astype(np.float64)
    add("opp_dist", np.where(okp, od, 0.0))
    add("opp_relspeed", np.where(okp & np.isfinite(rs), rs, 0.0))
    # ---- time since the ball's last direction change
    add("ball_since_turn", np.log1p(fe.since_last(seen & (turn >= TURN_THR), 60 * fps)))
    E = np.stack(cols, 1).astype(np.float32)
    return E, names, kinds
