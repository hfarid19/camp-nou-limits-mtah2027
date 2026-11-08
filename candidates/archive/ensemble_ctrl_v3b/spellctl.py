"""Spell-level "controlled change" features and the keep-or-hand-back rule (ensemble_ctrl_v3b).

Shared by the live pipeline and the offline fit (train.py), so both compute identical features.
Everything here reads only the degraded inputs (players, slot_team, ball, sign, blocks) and the
ensemble's own outputs (the unfolded vote codes and the members' codes and P(H, A, D)). Never a label.

A *contested* spell is an H or A spell that directly follows the other team's spell inside a graded
block (not after D, not at a block start): exactly the spells control.fold may hand back.

apply() mirrors control.fold: every pass computes the segments and features of the current codes,
scores every contested spell, then walks left to right and hands a spell back to the preceding team
(out[a - 1], re-read as it changes) when P(keep) < thr. Repeated until nothing changes (iterate=True,
at most 10 passes) or once (iterate=False).
"""

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import savgol_filter

H, A, D, X = 0, 1, 2, 3
FPS = 25
PITCH = (105.0, 68.0)
RADII = (0.5, 1.0, 2.0, 3.0, 5.0)
TOUCH_ACC = 5.0         # m/s^2, Ball Radar / L&H kick peak
TOUCH_ANG = 10.0        # degrees, databallpy ba_threshold
TOUCH_R = 1.5           # m, databallpy pz_radius
DUEL_R = 1.0            # m, Vidal-Codina duel zone
FLIGHT_M = 15.0         # m, long flight without a player within TOUCH_R


def _runs_true(m):
    """Longest run of True in a bool array (frames)."""
    if not m.any():
        return 0
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((e - s).max())


class Frames(object):
    """Per-frame quantities of one game, computed once."""

    def __init__(self, players, slot_team, ball, sign_half1, blocks, member_codes, member_probs,
                 pitch=PITCH, fps=FPS):
        self.fps = fps
        self.pitch = pitch
        n = len(ball)
        self.n = n
        ball = np.asarray(ball, np.float64)
        seen = ~np.isnan(ball[:, 0])
        idx = np.arange(n)
        b = ball.copy()
        if seen.sum() >= 2:
            for k in (0, 1):
                b[:, k] = np.interp(idx, idx[seen], ball[seen, k])
        else:
            b[:] = np.array(pitch) / 2
        self.seen = seen
        self.b = b
        v = savgol_filter(b, 7, 2, deriv=1, axis=0) * fps
        acc = np.linalg.norm(savgol_filter(b, 7, 2, deriv=2, axis=0) * fps * fps, axis=1)
        sp = np.linalg.norm(v, axis=1)
        ang = np.zeros(n)
        u0, u1 = v[:-3], v[3:]
        cos = (u0 * u1).sum(1) / (np.linalg.norm(u0, axis=1) * np.linalg.norm(u1, axis=1) + 1e-9)
        ang[1:-2] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        seen7 = minimum_filter1d(seen.astype(np.int8), 7, mode="constant", cval=0) > 0
        self.seen7 = seen7
        self.speed = np.where(seen7, sp, np.nan)
        players = np.asarray(players, np.float32)
        st = np.asarray(slot_team)
        dist = np.linalg.norm(players - ball[:, None, :].astype(np.float32), axis=2)
        dist = np.where(np.isnan(dist), np.inf, dist)
        self.dmin = np.full((2, n), np.inf, np.float32)
        self.cnt = np.zeros((2, len(RADII), n), np.int8)
        for t in (0, 1):
            dt = dist[:, st == t]
            if dt.shape[1]:
                self.dmin[t] = dt.min(1)
                for i, r in enumerate(RADII):
                    self.cnt[t, i] = (dt < r).sum(1)
        jn = np.argmin(dist, 1)
        dn = dist[idx, jn]
        near_any = seen & (dn < TOUCH_R)
        # long flight: ball displacement / time since a player was last within TOUCH_R
        last = np.maximum.accumulate(np.where(near_any, idx, -1))
        has = last >= 0
        lastc = np.where(has, last, 0)
        self.flight_m = np.where(has & seen, np.linalg.norm(b - b[lastc], axis=1), 0.0)
        self.flight_s = np.where(has, (idx - lastc) / fps, 99.0)
        # touch events: acceleration / direction peaks with a player within TOUCH_R, one per burst
        cand = seen7 & near_any & ((acc > TOUCH_ACC) | ((ang > TOUCH_ANG) & (sp > 1.0)))
        team_n = np.where(cand, st[jn], -1)
        ev_f, ev_t, ev_s, ev_d = [], [], [], []
        cf = np.flatnonzero(cand)
        if len(cf):
            brk = np.flatnonzero((np.diff(cf) > 2) | (np.diff(team_n[cf]) != 0)) + 1
            for g in np.split(cf, brk):
                f = int(g[np.argmax(acc[g])])
                t = int(team_n[f])
                duel = bool(self.dmin[1 - t, f] < DUEL_R)
                ev_f.append(f); ev_t.append(t); ev_s.append(int(jn[f])); ev_d.append(duel)
        self.ev_f = np.array(ev_f, np.int64)
        self.ev_t = np.array(ev_t, np.int64)
        self.ev_s = np.array(ev_s, np.int64)
        self.ev_d = np.array(ev_d, bool)
        self.ev_pos = (players[self.ev_f, self.ev_s].astype(np.float64) if len(ev_f)
                       else np.zeros((0, 2)))
        # attack direction of home per frame: +1 towards x = L
        self.hdir = np.zeros(n, np.float64)
        for half, first, last_ in blocks:
            self.hdir[first:last_ + 1] = sign_half1 if int(half) == 1 else -sign_half1
        self.mc = [np.asarray(c, np.int8) for c in member_codes]
        self.mp = [np.asarray(p, np.float32).astype(np.float64) for p in member_probs]
        for k, p in enumerate(self.mp):
            p = np.nan_to_num(p, nan=1.0 / 3)
            p = np.clip(p, 0.0, None)
            s = p.sum(1, keepdims=True)
            s[s <= 0] = 1.0
            self.mp[k] = p / s
        self._cache = {}


FEATURES = (
    ["dur_s", "prev_dur_s", "next_kind", "next_dur_s", "pp_same", "pp_dur_s",
     "m0_c", "m1_c", "m2_c", "m_c", "m0_p", "m1_p", "m2_p", "all3_c", "le1_c",
     "pm_c", "pm_p", "pm0_c", "pm1_c", "pm2_c", "pm_c_start", "pm_p_start",
     "vis", "vis_start",
     "tc", "tp", "tduel", "tc_1s", "first_tc_s", "pass_c", "n_slots_c", "last_touch_before",
     "since_tp_s",
     ] + ["opp_r{:g}".format(r) for r in RADII] + ["own_r1", "own_r2",
     "dmin_c", "dmin_p", "near_share_c", "pz_c", "pz_run_c", "pz_run_p",
     "duel_share", "contest_share",
     "v_in", "v_out", "v_mean", "v_max", "disp",
     "flight_m", "flight_s", "flight_flag",
     "dgoal", "touchline", "dgoal_end", "touchline_end",
     "regain_1s", "regain_2s"])


def _nanmean(x):
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def segments(codes, blocks):
    out = []
    for _, first, last in blocks:
        w = codes[first:last + 1]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bd = np.concatenate(([0], br, [len(w)]))
        seg = [(first + int(bd[i]), first + int(bd[i + 1]), int(w[bd[i]])) for i in range(len(bd) - 1)]
        out.append(seg)
    return out           # list per block


def spell_features(F, seg, i, codes):
    """Features of contested spell seg[i] (seg: one block's segments)."""
    a, b, c = seg[i]
    pa, _, p = seg[i - 1]
    nxt = seg[i + 1] if i + 1 < len(seg) else None
    pp = seg[i - 2] if i >= 2 else None
    key = (pa, a, b, c, nxt, pp, codes[b:b + 2 * F.fps].tobytes())
    if key in F._cache:
        return F._cache[key]
    fps = F.fps
    n = F.n
    f = {}
    f["dur_s"] = (b - a) / fps
    f["prev_dur_s"] = (a - pa) / fps
    if nxt is None:
        f["next_kind"], f["next_dur_s"] = 2, 0.0
    else:
        f["next_kind"] = 1 if nxt[2] == D else 0
        f["next_dur_s"] = (nxt[1] - nxt[0]) / fps
    f["pp_same"] = 1 if (pp is not None and pp[2] == c) else (0 if pp is not None else -1)
    f["pp_dur_s"] = (pp[1] - pp[0]) / fps if pp is not None else 0.0
    sl = slice(a, b)
    mc = [m[sl] for m in F.mc]
    for k in range(3):
        f["m{}_c".format(k)] = float((mc[k] == c).mean())
        f["m{}_p".format(k)] = float((mc[k] == p).mean())
    nc = sum((m == c).astype(np.int8) for m in mc)
    f["m_c"] = float(nc.mean()) / 3
    f["all3_c"] = float((nc == 3).mean())
    f["le1_c"] = float((nc <= 1).mean())
    pc = [pm[sl, c] for pm in F.mp]
    for k in range(3):
        f["pm{}_c".format(k)] = float(pc[k].mean())
    f["pm_c"] = float(np.mean([x.mean() for x in pc]))
    f["pm_p"] = float(np.mean([pm[sl, p].mean() for pm in F.mp]))
    s0 = slice(a, min(b, a + fps // 2))
    f["pm_c_start"] = float(np.mean([pm[s0, c].mean() for pm in F.mp]))
    f["pm_p_start"] = float(np.mean([pm[s0, p].mean() for pm in F.mp]))
    f["vis"] = float(F.seen[sl].mean())
    w0 = slice(max(0, a - fps // 2), min(n, a + fps // 2))
    f["vis_start"] = float(F.seen[w0].mean())
    # touches
    lo, hi = np.searchsorted(F.ev_f, [max(0, a - fps // 2), b])
    et, es, ed, ef = F.ev_t[lo:hi], F.ev_s[lo:hi], F.ev_d[lo:hi], F.ev_f[lo:hi]
    f["tc"] = int(((et == c) & ~ed).sum())
    f["tp"] = int(((et == p) & ~ed).sum())
    f["tduel"] = int(ed.sum())
    f["tc_1s"] = int(((et == c) & (ef < a + fps)).sum())
    fc = ef[et == c]
    f["first_tc_s"] = (fc[0] - a) / fps if len(fc) else np.nan
    # completed pass: consecutive touches (in [a - 0.5 s, b + 0.5 s]) both by c, different players
    lo2, hi2 = np.searchsorted(F.ev_f, [max(0, a - fps // 2), min(n, b + fps // 2)])
    pas = 0
    for k in range(lo2, hi2 - 1):
        if F.ev_t[k] == c and F.ev_t[k + 1] == c and F.ev_s[k] != F.ev_s[k + 1] and \
                np.linalg.norm(F.ev_pos[k] - F.ev_pos[k + 1]) > 3.0:
            pas += 1
    f["pass_c"] = pas
    f["n_slots_c"] = len(set(es[et == c].tolist()))
    k0 = np.searchsorted(F.ev_f, max(0, a - fps // 2)) - 1
    if k0 >= 0 and F.ev_f[k0] >= a - 3 * fps:
        f["last_touch_before"] = 1 if F.ev_t[k0] == c else 0
    else:
        f["last_touch_before"] = -1
    kp = np.flatnonzero(F.ev_t[:max(0, np.searchsorted(F.ev_f, a + fps // 2))] == p)
    f["since_tp_s"] = min(10.0, (a - F.ev_f[kp[-1]]) / fps) if len(kp) else 10.0
    # proximity at the start (seen frames of the first 0.5 s)
    e0 = np.arange(a, min(b, a + fps // 2))
    e0 = e0[F.seen[e0]]
    for ri, r in enumerate(RADII):
        f["opp_r{:g}".format(r)] = float(F.cnt[p, ri, e0].mean()) if len(e0) else np.nan
    f["own_r1"] = float(F.cnt[c, 1, e0].mean()) if len(e0) else np.nan
    f["own_r2"] = float(F.cnt[c, 2, e0].mean()) if len(e0) else np.nan
    sv = np.arange(a, b)[F.seen[sl]]
    if len(sv):
        dc, dp = F.dmin[c, sv], F.dmin[p, sv]
        f["dmin_c"] = float(np.minimum(dc, 30).mean())
        f["dmin_p"] = float(np.minimum(dp, 30).mean())
        f["near_share_c"] = float((dc < dp).mean())
        f["pz_c"] = float((dc < TOUCH_R).mean())
        f["duel_share"] = float((F.cnt[p, 1, sv] >= 2).mean())
        f["contest_share"] = float(((dc < DUEL_R) & (dp < DUEL_R)).mean())
    else:
        for k in ("dmin_c", "dmin_p", "near_share_c", "pz_c", "duel_share", "contest_share"):
            f[k] = np.nan
    pzc = F.seen[sl] & (F.dmin[c, sl] < TOUCH_R)
    pzp = F.seen[sl] & (F.dmin[p, sl] < TOUCH_R)
    f["pz_run_c"] = _runs_true(pzc) / fps
    f["pz_run_p"] = _runs_true(pzp) / fps
    f["v_in"] = _nanmean(F.speed[max(0, a - fps // 2):a])
    f["v_out"] = _nanmean(F.speed[max(a, b - fps // 4):min(n, b + fps // 2)])
    f["v_mean"] = _nanmean(F.speed[sl])
    vs = F.speed[sl]
    vs = vs[np.isfinite(vs)]
    f["v_max"] = float(vs.max()) if len(vs) else np.nan
    f["disp"] = float(np.linalg.norm(F.b[b - 1] - F.b[a]))
    w1 = slice(max(0, a - fps // 2), min(b, a + fps // 4) + 1)
    f["flight_m"] = float(F.flight_m[w1].max())
    f["flight_s"] = float(min(F.flight_s[a], 10.0))
    f["flight_flag"] = int(f["flight_m"] > FLIGHT_M)
    L, W = F.pitch

    def goal_d(t):
        d = F.hdir[t] if c == H else -F.hdir[t]
        xg = L if d > 0 else 0.0
        return float(np.hypot(F.b[t, 0] - xg, F.b[t, 1] - W / 2)), float(min(F.b[t, 1], W - F.b[t, 1]))
    f["dgoal"], f["touchline"] = goal_d(a)
    f["dgoal_end"], f["touchline_end"] = goal_d(b - 1)
    f["regain_1s"] = float((codes[b:min(n, b + fps)] == p).mean()) if b < n else 0.0
    f["regain_2s"] = float((codes[b:min(n, b + 2 * fps)] == p).mean()) if b < n else 0.0
    row = np.array([f[k] for k in FEATURES], np.float64)
    row = np.round(row, 4)
    F._cache[key] = row
    return row


def contested(codes, blocks):
    """[(block_index, seg_index, a, b, c)] contested spells and the per-block segments."""
    segs = segments(codes, blocks)
    out = []
    for bi, seg in enumerate(segs):
        for i in range(1, len(seg)):
            a, b, c = seg[i]
            p = seg[i - 1][2]
            if c in (H, A) and p in (H, A) and p != c:
                out.append((bi, i, a, b, c))
    return out, segs


def table(F, codes, blocks):
    """(spells [(a, b, c)], features (n, len(FEATURES))) of the contested spells of codes."""
    sp, segs = contested(codes, blocks)
    rows = [spell_features(F, segs[bi], i, codes) for bi, i, _, _, _ in sp]
    X_ = np.array(rows, np.float64) if rows else np.zeros((0, len(FEATURES)))
    return [(a, b, c) for _, _, a, b, c in sp], X_


def apply(F, codes, blocks, predict, thr, iterate=True, max_pass=10):
    """Hand back every contested spell with predict(X) < thr; control.fold's pass structure."""
    out = np.asarray(codes, np.int8).copy()
    for _ in range(max_pass if iterate else 1):
        spells, X_ = table(F, out, blocks)
        if not len(spells):
            break
        pk = predict(X_)
        changed = False
        for (a, b, c), q in zip(spells, pk):
            if q < thr and out[a - 1] in (H, A) and out[a - 1] != c:
                out[a:b] = out[a - 1]
                changed = True
        if not changed:
            break
    return out
