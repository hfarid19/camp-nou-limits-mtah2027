# ARCHIVE -- NOT the submission. Earlier revision kept for the README history; it may import
# sibling candidates or models that no longer exist in this form. The submission is ../../pipeline.py.
"""hier_touch_decoder: hierarchical two-stage possession decoder.

Stage A  alive/dead: 2-label minimum-duration Viterbi on a rebuilt dead-ball logit
         (S2 recipe), then asymmetric snaps of the D edges (D end -> end of a long ball gap,
         D start -> first ball-out frame in a 12 s look-back, else a long-gap start) and the
         kick-off pre-D forcing from Smart Tagging.
Stage B  team inside each alive run (run edges are hard boundaries): 2-label (H/A)
         minimum-duration Viterbi with a per-frame switch cost that drops near a confident-touch
         onset by the non-carrying team and at a team-velocity reversal. dmin is relaxed for the
         first and last segment of each run.
POST     H<->A snap to the new team's first touch (radius * k), prune spells with no own-team
         control frame, merge spells < 0.5 s.

Evidence: shape model, near-ball term, carry and the S2 dead logit are all computed locally
(numpy, pp helpers only). k = clip(q61(dmin | ball seen, players moving) / 1.5, 1, 2) scales
every radius: k = 1.0 on the bench, 1.83 on the real match (its ball sits ~4x farther from the feet).

TUNING (shared bench, dev = Sample Games 1+2 x 2 seeds; spellF1 mean / macroF1 mean;
"off" = AR(1) ball-offset sanity check, sd 1.3 m, tau 1 s, where v1 scores ~0.437)
  v1_original baseline                                              0.4749 / 0.7453
  stage A only (v1 3-state team decoding with D forced)             0.5332 / 0.8009
    c_dead {3..7} x dmin_D {1,2,3}s x dmin_alive {.5,1,2}s, a_temp {.08,.15,.3}, snaps on/off:
    all within 0.5322-0.5343 -> frozen at c_dead 5, dmin_D 2 s, dmin_alive 1 s, temp 0.15
  first hier version (spec defaults: sw 4, a_touch .85, a_vel .5, dmin 50, r2 2.0) 0.5768 / 0.8092
  coarse grid sw {3,4,6} x a_touch {.6,.85} x a_vel {0,.5,.8} x dmin {37,50,62} x w_carry {1,2}:
    best 0.5902 (sw 3, .6, .8, 37, 1); w_carry 1 >> 2; dmin 62 worst (~0.571)
  extended: sw {2,3,4} x a_vel {.5,.8} x dmin {19..44} x w_carry {.5,1,1.5}: flat 0.585-0.591
    for w_carry 1 (w_carry .5 mean 0.572, 1.5 mean 0.579)
  w_shape {.4,.7,1} x w_near {.5,1,2} x w_carry {.75,1,1.25} x temp {.1,.15,.25}: top 0.594,
    defaults (0.7, 1, 1, 0.15) at 0.590 -> kept (within noise)
  touch r {.7,.9,1.1} x r2 {1,1.25,1.5,1.75,2} x len {1,2,3}: clean best 0.621 (r2 1.0, len 1)
    but that drops to 0.489 under "off"; chosen r .9, r2 1.25, len 3 = 0.6032 clean / 0.5185 off
    (best clean+off sum; neighbours within 0.01 clean)
  re-grid at that touch point, sw {2,3,4} x a_touch {.6,.85} x a_vel {.5,.8} x dmin {25,37,50}:
    plateau sw 3-4, a_touch .6, a_vel .5-.8, dmin 37-50 -> sw 4, .6, .8, 37:
                                                        FINAL    0.6035 / 0.8121  (off 0.5202)
    rows 0.5599 / 0.6056 / 0.6202 / 0.6285, all above v1 (0.5002 / 0.4858 / 0.4591 / 0.4546)
  stage A re-check with the final B: still flat (0.6038-0.6040)
  ablations from FINAL: a_vel 0 0.5941 | a_touch 0 0.5849 | both 0 0.5745 | w_carry 0 0.5521 |
    w_near 0 0.5735 | w_shape 0 0.5847 | prune off 0.6018 | team snap off 0.6028 |
    min_spell 0 0.6040 | dmin_HA 1 0.6003 | D-edge snaps off 0.6013 |
    v1 dead logit (slow 3.5, long 1.0 ungated) 0.5662
  near_r / snap_r / snap_win {1,1.5,2.5}^3: all within 0.006 of FINAL -> defaults kept
  sandwich states (short H|A interceptions, near+touch emission, <=1.5 s): sand_c .25 0.546,
    .5 0.562, 1.0 0.590, 2.0 0.602, 3.0 0.6044 (Game 2 down) -> REJECTED, left in, off
  STRETCH semi-Markov DP on a touch lattice (loose 1.5k onsets + run edges, lognormal duration
    prior median 5.5 s sigma 1.3, touch-count term): alpha {0,.25,.5,1} x w_dur {0,.25,.5,1} x
    sw {2,4,6,10}: best 0.5719 -> REJECTED (code removed)
  real match (--match): 269 possessions, dead share 44.9%, k 1.83; byte-identical across runs.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import possession_pipeline as pp  # noqa: E402

H, A, D, X = 0, 1, 2, 3

CFG = dict(
    # ---- shared evidence
    gap_fill_s=0.5,
    dead_speed_ms=1.4,
    min_tracked=12,
    k_quantile=61.0,      # match-adaptive radius scale k = clip(q(dmin | seen, moving) / 1.5, 1, 2)
    k_lo=1.0, k_hi=2.0,
    # ---- stage A (dead logit, S2 recipe)
    a_bias=-1.5, a_out=4.0, a_slow=2.0, a_few=4.0, a_long=4.0, a_static=0.8, w_tag_dead=1.0,
    long_s=3.0, lm_gate=2.0,
    a_temp=0.15,
    c_dead=5.0,           # alive->D and D->alive transition cost
    dmin_D_s=2.0, dmin_alive_s=1.0,
    snap_gap_s=4.0, look_back_s=12.0, keep_s=0.5,
    prekick=1,
    # ---- stage B
    b_mode="hier",        # "hier" (this candidate) or "v1" (v1 3-state run with D forced: stage-A tuning)
    w_shape=0.7, w_near=1.0, w_carry=1.0, w_tag=0.8,
    near_r=1.5,           # metres (x k) gate and scale of the near-ball term
    touch_r=0.9, touch_r2=1.25, touch_len=3,
    b_temp=0.15,
    base_sw=4.0, a_touch=0.6, a_vel=0.8, sw_floor=0.2, onset_w=3,
    vel_hw=12, vel_lag=25,
    dmin_HA=37,
    sandwich=0, sand_s=1.5, sand_c=0.5, w_stouch=1.0,   # optional short-interception states
    # ---- post
    snap_r=1.5, snap_win_s=1.5,
    prune=1,
    min_spell_s=0.5,
)
CFG.update(json.loads(os.environ.get("HTD_CFG", "{}")))


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def runs_bool(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def segments(path):
    c = np.concatenate(([0], np.flatnonzero(np.diff(path)) + 1, [len(path)]))
    return [(int(c[i]), int(c[i + 1]), int(path[c[i]])) for i in range(len(c) - 1)]


def logsig(x):
    return -np.logaddexp(0.0, -x)


def mindur_viterbi2(E, dmin, enter_cost, relax_last=False, sand=None):
    """Exact 2-label Viterbi with a minimum segment duration (semi-Markov form of the
    expanded-state chain). E (T, 2) log-emissions; dmin (d0, d1) frames; enter_cost (T, 2): cost
    of a segment of label l starting at frame s (charged once, at the change). The first segment
    is free of dmin; the last one too when relax_last.

    sand = (Es (T, 2), cost (T,), L): optional 'sandwich' states. Inside a long run of label l a
    short spell (1..L frames) of the other label may be inserted, scored with Es instead of E,
    entered at cost[s] and left back to l for free. Kept exact with a sliding-window maximum."""
    from collections import deque
    T = len(E)
    if T == 0:
        return np.zeros(0, np.int8)
    Pc = (np.concatenate(([0.0], np.cumsum(E[:, 0]))).tolist(),
          np.concatenate(([0.0], np.cumsum(E[:, 1]))).tolist())
    ee = (E[:, 0].tolist(), E[:, 1].tolist())
    cst = (enter_cost[:, 0].tolist(), enter_cost[:, 1].tolist())
    dd = (int(max(1, dmin[0])), int(max(1, dmin[1])))
    f = ([0.0] * T, [0.0] * T)
    bp = ([-1] * T, [-1] * T)
    f[0][0], f[1][0] = ee[0][0], ee[1][0]
    if sand is not None:
        Es, csh, L = sand
        Qs = (np.concatenate(([0.0], np.cumsum(Es[:, 0]))).tolist(),
              np.concatenate(([0.0], np.cumsum(Es[:, 1]))).tolist())
        csh = csh.tolist()
        L = int(L)
        dq = (deque(), deque())       # (g, start) with decreasing g
    for t in range(1, T):
        for l in (0, 1):
            o = 1 - l
            fl, fo, P = f[l], f[o], Pc[l]
            v = fl[t - 1] + ee[l][t]
            b = -1
            s = t - dd[l] + 1
            if s >= 1:
                w = fo[s - 1] - cst[l][s] + P[t + 1] - P[s]
                if w > v:
                    v, b = w, s
            if sand is not None:
                q = dq[l]
                Q = Qs[o]
                g = fl[t - 1] - csh[t] - Q[t]
                while q and q[-1][0] <= g:
                    q.pop()
                q.append((g, t))
                while q[0][1] < t - L + 1:
                    q.popleft()
                w = q[0][0] + Q[t + 1]
                if w > v:
                    v, b = w, -(q[0][1] + 2)
            fl[t] = v
            bp[l][t] = b
    best, lab, start = f[0][T - 1], 0, -1
    if f[1][T - 1] > best:
        best, lab = f[1][T - 1], 1
    if relax_last:
        for l in (0, 1):
            o = f[1 - l]
            for s in range(max(1, T - dd[l] + 1), T):
                w = o[s - 1] - cst[l][s] + Pc[l][T] - Pc[l][s]
                if w > best:
                    best, lab, start = w, l, s
    path = np.empty(T, np.int8)
    t = T - 1
    if start >= 0:
        path[start:] = lab
        t, lab = start - 1, 1 - lab
    while t >= 0:
        bl = bp[lab]
        s = t
        while s > 0 and bl[s] == -1:
            s -= 1
        if bl[s] == -1:            # reached frame 0 inside the first segment
            path[:t + 1] = lab
            break
        if bl[s] <= -2:            # sandwich [ss, s] of the other label inside lab
            ss = -bl[s] - 2
            path[s + 1:t + 1] = lab
            path[ss:s + 1] = 1 - lab
            t = ss - 1
            continue
        st = bl[s]
        path[st:t + 1] = lab
        t, lab = st - 1, 1 - lab
    return path


# --------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------

def side_of(team, names):
    home_name, away_name = names[0].lower(), names[1].lower()
    if not team or team in ("n/a", "na"):
        return 0.0
    if team == home_name or "barc" in team:
        return 1.0
    if team == away_name or "atl" in team or "athl" in team:
        return -1.0
    return 0.0


def restart_after(raw_seen, s, horizon):
    """First frame >= s where the ball is seen again after being missing (ball back in play)."""
    w = raw_seen[s:s + horizon]
    miss = np.where(~w)[0]
    if not len(miss):
        return s + int(0.15 * horizon)
    after = np.where(w[miss[0]:])[0]
    return s + miss[0] + int(after[0]) if len(after) else s + int(0.15 * horizon)


def features(players, slot_team, ball_raw, sign_half1, blocks, tags, names, pitch, fps, C=None):
    C = C or CFG
    length, width = pitch
    n = len(ball_raw)
    home, away = slot_team == 0, slot_team == 1
    sign = np.full(n, float(sign_half1))
    for half, first, last in blocks:
        if half == 2:
            sign[first:] = -sign_half1
    gap = int(round(C["gap_fill_s"] * fps))
    ball = np.stack([pp.fill_short_gaps(ball_raw[:, 0].astype(np.float64), gap),
                     pp.fill_short_gaps(ball_raw[:, 1].astype(np.float64), gap)], 1)
    raw_seen = ~np.isnan(ball_raw[:, 0])
    seen = ~np.isnan(ball[:, 0])
    dist = np.sqrt(((players - ball[:, None, :].astype(np.float32)) ** 2).sum(-1))
    d_home = np.fmin.reduce(dist[:, home], axis=1).astype(np.float64)
    d_away = np.fmin.reduce(dist[:, away], axis=1).astype(np.float64)
    del dist
    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * fps
    spd = np.sqrt((vel ** 2).sum(-1))
    spd[spd > 12.0] = np.nan
    mean_speed = pp.moving_mean(np.nanmean(spd, axis=1), fps)
    n_tracked = pp.moving_mean((~np.isnan(players[:, :, 0])).sum(1).astype(np.float64), fps)
    bvel = np.full(n, np.nan)
    bvel[1:] = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(1)) * fps
    ball_speed = pp.moving_mean(bvel, fps // 2)

    # match-adaptive radius scale (v2 S5)
    dmin = np.fmin(d_home, d_away)
    moving = np.nan_to_num(mean_speed, nan=0.0) > C["dead_speed_ms"]
    sel = raw_seen & moving & ~np.isnan(dmin)
    q = float(np.percentile(dmin[sel], C["k_quantile"])) if sel.sum() > 100 else 1.5
    k = float(np.clip(q / 1.5, C["k_lo"], C["k_hi"]))

    # team shape model (copy of pp.build_evidence, pseudo-label radius x k)
    x_dir = (players[:, :, 0] - length / 2.0) * sign[:, None]

    def team_stats(mask, s):
        xs, ys = x_dir[:, mask] * s, players[:, mask, 1]
        vx = vel[:, mask, 0] * sign[:, None] * s
        return (np.nanmean(xs, 1), np.nanstd(xs, 1) + np.nanstd(ys, 1),
                np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1),
                np.nanmean(np.clip(vx, -10, 10), 1))
    hc, hs, hl, hw, hv = team_stats(home, 1.0)
    ac, as_, al, aw, av = team_stats(away, -1.0)
    raw_feats = [np.nanmean(x_dir, 1), hc + ac, hs - as_, hl - al, hw - aw, hv - av]
    F = np.stack([pp.moving_mean(f, fps) for f in raw_feats], 1)
    F_ok = ~np.isnan(F).any(1)
    r = 1.5 * k
    alive_like = mean_speed > C["dead_speed_ms"]
    ph_home = raw_seen & (d_home < r) & (d_away > 2 * r) & alive_like & F_ok
    ph_away = raw_seen & (d_away < r) & (d_home > 2 * r) & alive_like & F_ok
    idx = np.where(ph_home | ph_away)[0]
    p_shape = np.full(n, 0.5)
    if len(idx) > 500 and ph_home.sum() > 100 and ph_away.sum() > 100:
        mu, sd = F[idx].mean(0), F[idx].std(0) + 1e-6
        Z = (F[idx] - mu) / sd
        w, b = pp.fit_logistic(Z, ph_home[idx].astype(float))
        Zall = (np.where(F_ok[:, None], F, mu) - mu) / sd
        p_shape = np.where(F_ok, pp.sigmoid(Zall @ w + b), 0.5)
    p_shape = np.clip(p_shape, 1e-4, 1 - 1e-4)
    l_shape = np.clip(np.log(p_shape / (1 - p_shape)), -3, 3)

    # velocity-reversal cue (probe_dyn)
    m = pp.moving_mean(hv - av, int(C["vel_hw"]))
    lag = int(C["vel_lag"])
    prev = np.r_[np.full(lag, np.nan), m[:-lag]]
    nxt = np.r_[m[lag:], np.full(lag, np.nan)]
    rev = np.nan_to_num(np.maximum(0.0, -(prev * nxt)))

    # dead-ball cues (S2 recipe)
    mg = 0.5
    out = seen & ((ball[:, 0] < -mg) | (ball[:, 0] > length + mg) |
                  (ball[:, 1] < -mg) | (ball[:, 1] > width + mg))
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    slow = pp.sigmoid((C["dead_speed_ms"] - np.nan_to_num(mean_speed, nan=0.0)) / 0.3)
    few = np.nan_to_num(n_tracked, nan=0.0) < C["min_tracked"]
    lm = np.zeros(n, bool)
    gaps = []
    for s, e in zip(*runs_bool(~seen)):
        if e - s > C["long_s"] * fps:
            gaps.append((int(s), int(e)))
            if np.nanmean(mean_speed[s:e]) < C["lm_gate"]:
                lm[s:e] = True
    static = seen & (np.nan_to_num(ball_speed, nan=9.0) < 0.3)

    # Smart Tagging (corrected windows, v2 S7)
    tag_dead = np.zeros(n)
    tag_team = np.zeros(n)
    restarts = []
    for code, s, e, team, kind in tags:
        s, e = max(0, s), min(n, max(e, s + 1))
        t = side_of(team, names)
        if code in pp.POSSESSING_CODES:
            tag_team[s:e] += t
        elif code in pp.DEFENDING_CODES:
            tag_team[s:e] -= t
        elif code == "set pieces":
            rs = restart_after(raw_seen, s, 20 * fps)
            tag_dead[max(0, s - fps):rs] += 1
            tag_team[rs:e] += t
            if "kick off (start)" in kind:
                restarts.append(rs)
        elif code == "goals":
            ends = [g for g in gaps if abs(g[1] - e) < 15 * fps]
            if ends:
                g0 = min(ends, key=lambda g: abs(g[1] - e))
                tag_dead[g0[0]:g0[1]] += 1
    tag_dead = np.clip(tag_dead, 0, 1)
    tag_team = np.clip(tag_team, -1, 1)

    return dict(n=n, fps=fps, k=k, d_home=d_home, d_away=d_away, raw_seen=raw_seen, seen=seen,
                out=out, slow=slow, few=few, lm=lm, static=static, gaps=gaps,
                tag_dead=tag_dead, tag_team=tag_team, restarts=restarts, l_shape=l_shape, rev=rev,
                v1=(players, ball_raw, slot_team, sign_half1, blocks, tags, names, pitch))


# --------------------------------------------------------------------------------------------
# stage A
# --------------------------------------------------------------------------------------------

def dead_logit(Fe, C):
    return (C["a_bias"] + C["a_out"] * Fe["out"] + C["a_slow"] * (2 * Fe["slow"] - 1)
            + C["a_few"] * Fe["few"] + C["a_long"] * Fe["lm"] + C["a_static"] * Fe["static"]
            + C["w_tag_dead"] * Fe["tag_dead"])


def stage_a(Fe, first, last, C, logit):
    """Dead mask (bool) over the block [first, last]."""
    fps = Fe["fps"]
    seg = slice(first, last + 1)
    T = last + 1 - first
    E = np.zeros((T, 2))
    E[:, 1] = C["a_temp"] * logit[seg]
    cost = np.full((T, 2), C["c_dead"])
    dmin = (int(round(C["dmin_alive_s"] * fps)), int(round(C["dmin_D_s"] * fps)))
    dead = mindur_viterbi2(E, dmin, cost).astype(bool)
    path = np.where(dead, D, H).astype(np.int8)          # H = 'alive' placeholder here
    out = Fe["out"][seg]
    gaps = [(s - first, e - first) for s, e in Fe["gaps"] if s >= first and e <= last + 1]
    gs = np.array([g[0] for g in gaps]); ge = np.array([g[1] for g in gaps])
    w = int(C["snap_gap_s"] * fps)
    segs = segments(path)
    for i, (s, e, lab) in enumerate(segs):
        if lab != D:
            continue
        # D end -> end of a long ball gap within the window
        if e < T and len(ge):
            j = int(np.argmin(np.abs(ge - e)))
            if abs(ge[j] - e) <= w and ge[j] > s:
                tt = int(ge[j])
                if tt > e:
                    path[e:min(tt, T)] = D
                else:
                    path[tt:e] = H
        # D start -> first ball-out frame in the look-back, else a long-gap start
        if s > 0:
            s0 = segs[i - 1][0]
            lo = max(s0 + int(C["keep_s"] * fps), s - int(C["look_back_s"] * fps))
            hits = np.where(out[lo:s + fps])[0] if lo < s else np.zeros(0, int)
            if len(hits) and lo + int(hits[0]) < s:
                path[lo + int(hits[0]):s] = D
            elif len(gs):
                j = int(np.argmin(np.abs(gs - s)))
                if abs(gs[j] - s) <= w and gs[j] < e:
                    tt = int(gs[j])
                    if tt < s:
                        path[max(tt, lo if lo < s else tt):s] = D
                    else:
                        path[s:tt] = H
    if C["prekick"]:
        ko = [r for r in Fe["restarts"] if first <= r <= first + 10 * 60 * fps]
        if ko:
            path[:ko[0] - first] = D
    return path == D


# --------------------------------------------------------------------------------------------
# stage B
# --------------------------------------------------------------------------------------------

def team_inputs(Fe, C):
    k = Fe["k"]
    dh = np.nan_to_num(Fe["d_home"], nan=99.0)
    da = np.nan_to_num(Fe["d_away"], nan=99.0)
    dmin = np.fmin(dh, da)
    rn = C["near_r"] * k
    near = Fe["seen"] & (dmin < rn)
    l_near = np.where(near, np.clip((da - dh) / rn, -2, 2), 0.0)
    rs = Fe["raw_seen"]
    th = rs & (dh < C["touch_r"] * k) & (da > C["touch_r2"] * k)
    ta = rs & (da < C["touch_r"] * k) & (dh > C["touch_r2"] * k)
    touch = np.zeros(Fe["n"], np.int8)
    onsets = []
    for val, msk in ((1, th), (-1, ta)):
        for s, e in zip(*runs_bool(msk)):
            if e - s >= C["touch_len"]:
                touch[s:e] = val
                onsets.append((int(s), val))
    onsets.sort()
    base = C["w_shape"] * Fe["l_shape"] + C["w_near"] * l_near + C["w_tag"] * Fe["tag_team"]
    return base, C["w_near"] * l_near, touch, onsets


def stage_b_run(s0, e0, base, sand_base, touch, onsets_arr, Fe, C):
    """Team path (H/A) for the alive run [s0, e0)."""
    T = e0 - s0
    tr = touch[s0:e0]
    # carry: forward-filled last confident touch, reset at the run start
    idx = np.where(tr != 0, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    carry = np.where(idx >= 0, tr[np.maximum(idx, 0)], 0).astype(np.float64)
    e = base[s0:e0] + C["w_carry"] * carry
    # onset of a confident touch by the team not holding carry
    onset = np.zeros(T)
    lo, hi = np.searchsorted(onsets_arr[:, 0], [s0, e0])
    ow = int(C["onset_w"])
    for s, val in onsets_arr[lo:hi]:
        s = int(s) - s0
        before = carry[s - 1] if s > 0 else 0.0
        if before != val:
            onset[max(0, s - ow):min(T, s + ow + 1)] = 1.0
    sw = C["base_sw"] * (1 - C["a_touch"] * onset) * (1 - C["a_vel"] * np.tanh(Fe["rev"][s0:e0]))
    sw = np.maximum(sw, C["sw_floor"])
    tb = C["b_temp"]
    E = np.stack([tb * logsig(e), tb * logsig(-e)], 1)
    dm = int(C["dmin_HA"])
    sand = None
    if C["sandwich"]:
        es = sand_base[s0:e0] + C["w_stouch"] * tr
        Es = np.stack([tb * logsig(es), tb * logsig(-es)], 1)
        sand = (Es, C["sand_c"] * sw, int(round(C["sand_s"] * Fe["fps"])))
    path = mindur_viterbi2(E, (dm, dm), np.stack([sw, sw], 1), relax_last=True, sand=sand)
    return path


def stage_b_v1(Fe, first, last, dead, C):
    """v1's 3-state run with D forced by stage A (used to tune stage A alone)."""
    if "_v1E" not in Fe:
        P = dict(pp.PARAMS)
        Fe["_v1E"] = pp.build_evidence(*Fe["v1"], P, lambda *a, **k: None)
    E, aux = Fe["_v1E"]
    P = dict(pp.PARAMS)
    seg = slice(first, last + 1)
    Ex = E[seg].copy()
    Ex[dead, 0] = Ex[dead, 1] = -1e6
    Ex[~dead, 2] = -1e6
    path = pp.viterbi(Ex, P)
    path = pp.merge_short(path, P)
    path = pp.snap_boundaries(path, {k: v[seg] for k, v in aux.items()}, P)
    return path


# --------------------------------------------------------------------------------------------
# post
# --------------------------------------------------------------------------------------------

def snap_team(path, dh, da, seen, C, fps, k):
    """pp.snap_boundaries' H<->A rule: move the change to the first frame in +-window where the
    new team is within control radius (x k) and not farther than the old team."""
    path = path.copy()
    w = int(C["snap_win_s"] * fps)
    r = C["snap_r"] * k
    segs = segments(path)
    for i in range(1, len(segs)):
        s_prev, _, lab_prev = segs[i - 1]
        s, e, lab = segs[i]
        if lab_prev == D or lab == D:
            continue
        lo, hi = max(s_prev + 1, s - w), min(e - 1, s + w)
        if hi <= lo:
            continue
        d_new = dh if lab == H else da
        d_old = da if lab == H else dh
        ok = seen[lo:hi] & (d_new[lo:hi] < r) & ~(d_old[lo:hi] < d_new[lo:hi])
        hits = np.where(ok)[0]
        if len(hits):
            t = lo + int(hits[0])
            path[min(t, s):max(t, s)] = lab if t < s else lab_prev
    return path


def prune_uncontrolled(path, dh, da, seen, r, rounds=3):
    path = path.copy()
    dh = np.nan_to_num(dh, nan=99.0)
    da = np.nan_to_num(da, nan=99.0)
    for _ in range(rounds):
        segs = segments(path)
        changed = False
        for i, (a, b, lab) in enumerate(segs):
            if lab == D:
                continue
            own, oth = (dh, da) if lab == H else (da, dh)
            if (seen[a:b] & (own[a:b] < r) & (own[a:b] < oth[a:b])).any():
                continue
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                path[a:b] = prev
            elif prev in (H, A) and nxt in (H, A):
                path[a:b] = nxt
            elif prev in (H, A):
                path[a:b] = prev
            elif nxt in (H, A):
                path[a:b] = nxt
            else:
                continue          # isolated restart between two stoppages: keep
            changed = True
        if not changed:
            break
    return path


def merge_short_v2(path, min_len):
    """Absorb spells shorter than min_len (per label), shortest first, recomputing segments."""
    path = path.copy()
    for _ in range(100000):
        segs = segments(path)
        if len(segs) == 1:
            break
        cand = sorted((e - s, i) for i, (s, e, lab) in enumerate(segs) if e - s < min_len[lab])
        done = False
        for _, i in cand:
            s, e, lab = segs[i]
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                path[s:e] = prev
                done = True
                break
            if lab != D and prev == D and nxt == D:
                continue
            if prev is not None and nxt is not None and prev != D and nxt != D:
                path[s:e] = nxt
                done = True
                break
        if not done:
            break
    return path


# --------------------------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------------------------

def decode(Fe, blocks, C=None):
    C = C or CFG
    fps = Fe["fps"]
    k = Fe["k"]
    logit = dead_logit(Fe, C)
    out = np.full(Fe["n"], X, np.int8)
    if C["b_mode"] != "v1":
        base, sand_base, touch, onsets = team_inputs(Fe, C)
        onsets_arr = np.array(onsets, np.int64).reshape(-1, 2)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        dead = stage_a(Fe, first, last, C, logit)
        if C["b_mode"] == "v1":
            out[seg] = stage_b_v1(Fe, first, last, dead, C)
            continue
        path = np.full(last + 1 - first, D, np.int8)
        for s, e in zip(*runs_bool(~dead)):
            path[s:e] = stage_b_run(first + s, first + e, base, sand_base, touch, onsets_arr,
                                    Fe, C)
        dh, da = Fe["d_home"][seg], Fe["d_away"][seg]
        seen = Fe["seen"][seg]
        path = snap_team(path, dh, da, seen, C, fps, k)
        if C["prune"]:
            path = prune_uncontrolled(path, dh, da, seen, 1.5 * k)
        ms = int(round(C["min_spell_s"] * fps))
        path = merge_short_v2(path, {H: ms, A: ms, D: ms})
        out[seg] = path
    return out


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    Fe = features(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    return decode(Fe, blocks, CFG)
