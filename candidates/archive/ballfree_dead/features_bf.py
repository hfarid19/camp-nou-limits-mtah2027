"""Ball-free game-state features: players only, never the ball.

compute(players, slot_team, pitch, fps) -> (F float32 (N, K), names)

players (N, S, 2) metres NaN-missing (anonymous, fragmented slots), slot_team (S,) 0 home 1 away.
Every feature is
  * identity-free: counts, means and order statistics over a team's tracked slots;
  * team-swap invariant: a per-team quantity enters as the (min, max) of the two teams, so the
    dead/alive model needs no swap augmentation and no attack direction;
  * pitch-mirror invariant: positions enter as distances to the halfway line / centre line,
    to the nearest box / corner flag / touchline / goal line.

Groups (after Lang et al. 2022, Sci Rep 12:16291, who predict ball in/out of play from player
positions, velocities and accelerations with +-0.4/0.8/2/4/20/40/80 s temporal context; their
per-player inputs need identities, so here the same context is applied to team aggregates):
  kinematics   speed mean / median / p90, walking / stationary / sprinting shares, mean
               acceleration, velocity coherence |mean v| / mean |v|, collective acceleration
  shape        centroid distance from halfway / centre line, x / y spread, length, width
  set pieces   kick-off (teams split by the halfway line, players in the centre circle, someone
               on the spot), corner (players near a flag, near the goal line), throw-in (players
               on / beyond the touchline), box crowding (most players in one penalty area, the
               opposition in it too), a wall (max same-team players within 1.5 m of a player),
               tight marking (nearest-opponent distance), goal kick / penalty proxies (box
               emptiness with players just outside it)
  tracking     tracked-player counts (the competition's tracking collapses in stoppages)
  temporal     centred means over 1 / 4 / 10 / 30 s, past / future 2 s and 8 s means and their
               difference (restart = collective acceleration after a still phase), past / future
               5 s min / max of the mean speed, and Lang's shifted copies (+-0.4 ... 80 s) of a
               1 s-smoothed core subset.
"""

import numpy as np

CORE_SHIFT_S = (0.4, 0.8, 2.0, 4.0, 20.0, 40.0, 80.0)      # Lang et al. 2022 offsets
N_BASE = 49     # the first N_BASE columns are per-frame (no temporal context): the TCN input


def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def mmean(values, lo_off, hi_off):
    """NaN-aware mean of values[t+lo_off : t+hi_off+1], per t."""
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


def shift(x, k):
    y = np.full_like(x, np.nan)
    if k > 0:
        y[:-k] = x[k:]
    elif k < 0:
        y[-k:] = x[:k]
    else:
        y[:] = x
    return y


def _filled(x):
    """NaNs replaced by linear interpolation (edges held), for min / max filters."""
    x = np.asarray(x, np.float64)
    good = ~np.isnan(x)
    if not good.any():
        return np.zeros_like(x)
    i = np.arange(len(x))
    return np.interp(i, i[good], x[good])


def _roll_ext(x, lo_off, hi_off, fn):
    """(approximately) min / max of x[t+lo_off : t+hi_off+1]: a centred filter, shifted."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    h = (hi_off - lo_off) // 2
    f = minimum_filter1d if fn == "min" else maximum_filter1d
    out = f(_filled(x), size=2 * h + 1, mode="nearest")
    return shift(out, (lo_off + hi_off) // 2)


def _nanq_sorted(s, cnt, q):
    """q-quantile per row of s (rows sorted ascending, NaN at the end), cnt valid per row."""
    k = np.clip(np.floor(q * (cnt - 1)).astype(np.int64), 0, s.shape[1] - 1)
    v = np.take_along_axis(s, k[:, None], 1)[:, 0]
    return np.where(cnt > 0, v, np.nan)


def compute(players, slot_team, pitch, fps):
    L, W = float(pitch[0]), float(pitch[1])
    n = len(players)
    P = players.astype(np.float32)
    x, y = P[:, :, 0], P[:, :, 1]
    valid = ~np.isnan(x)
    teams = [slot_team == 0, slot_team == 1]
    cols, names = [], []

    def add(name, values):
        cols.append(np.asarray(values, np.float32))
        names.append(name)

    def pair(name, fh, fa):
        """team-swap invariant pair: min and max of the two teams' values."""
        add(name + "_min", np.fmin(fh, fa))
        add(name + "_max", np.fmax(fh, fa))

    # --- per-slot kinematics (central differences over +-2 frames inside a tracklet) ---
    k = 2
    v = np.full(P.shape, np.nan, np.float32)
    v[k:-k] = (P[2 * k:] - P[:-2 * k]) * (fps / (2.0 * k))
    spd = np.sqrt((v ** 2).sum(-1))
    bad = spd > 12.0
    spd[bad] = np.nan
    v[bad] = np.nan
    acc = np.full(spd.shape, np.nan, np.float32)
    acc[k:-k] = np.sqrt(((v[2 * k:] - v[:-2 * k]) ** 2).sum(-1)) * (fps / (2.0 * k))
    acc[acc > 15.0] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        cnt_t, mspd_t = [], []
        for ti, tm in enumerate(teams):
            s = spd[:, tm]
            c = (~np.isnan(s)).sum(1).astype(np.float64)
            cnt_t.append((valid[:, tm]).sum(1).astype(np.float64))
            m = np.nanmean(s, 1)
            mspd_t.append(m)
        cnt_all = cnt_t[0] + cnt_t[1]
        add("n_tracked", cnt_all)
        pair("n_team", cnt_t[0], cnt_t[1])
        sall = np.sort(spd, 1)
        call = (~np.isnan(spd)).sum(1)
        mean_speed = np.nanmean(spd, 1)
        add("speed_mean", mean_speed)
        add("speed_med", _nanq_sorted(sall, call, 0.5))
        add("speed_p90", _nanq_sorted(sall, call, 0.9))
        add("speed_max", _nanq_sorted(sall, call, 1.0))
        cden = np.maximum(call, 1)
        walk = np.where(call > 0, (spd < 2.0).sum(1) / cden, np.nan)
        still = np.where(call > 0, (spd < 0.7).sum(1) / cden, np.nan)
        sprint = np.where(call > 0, (spd > 4.0).sum(1) / cden, np.nan)
        add("walk_share", walk)
        add("still_share", still)
        add("run_share", sprint)
        acc_mean = np.nanmean(acc, 1)
        add("acc_mean", acc_mean)
        pair("speed_team", mspd_t[0], mspd_t[1])
        coh_t = []
        for tm in teams:
            mv = np.nanmean(v[:, tm], 1)
            ms = np.nanmean(spd[:, tm], 1)
            coh_t.append(np.sqrt((mv ** 2).sum(-1)) / np.maximum(ms, 0.3))
        pair("coherence", coh_t[0], coh_t[1])
        mv = np.nanmean(v, 1)
        coh_all = np.sqrt((mv ** 2).sum(-1)) / np.maximum(mean_speed, 0.3)
        add("coherence_all", coh_all)

        # --- team shape (mirror invariant) ---
        cx_t, sx_t, sy_t, fl_t = [], [], [], []
        for tm in teams:
            xt, yt = x[:, tm], y[:, tm]
            cx = np.nanmean(xt, 1)
            cx_t.append(cx)
            sx_t.append(np.nanstd(xt, 1))
            sy_t.append(np.nanstd(yt, 1))
            fl_t.append(np.where(valid[:, tm].sum(1) > 0,
                                 (xt < L / 2).sum(1) / np.maximum(valid[:, tm].sum(1), 1), np.nan))
        pair("cx_half", np.abs(cx_t[0] - L / 2), np.abs(cx_t[1] - L / 2))
        pair("spread_x", sx_t[0], sx_t[1])
        pair("spread_y", sy_t[0], sy_t[1])
        add("cx_gap", np.abs(cx_t[0] - cx_t[1]))
        cxa = np.nanmean(x, 1)
        cya = np.nanmean(y, 1)
        add("cx_all_half", np.abs(cxa - L / 2))
        add("cy_all_mid", np.abs(cya - W / 2))
        xs = np.sort(x, 1)
        ys = np.sort(y, 1)
        cv_ = valid.sum(1)
        add("length", _nanq_sorted(xs, cv_, 0.9) - _nanq_sorted(xs, cv_, 0.1))
        add("width", _nanq_sorted(ys, cv_, 0.9) - _nanq_sorted(ys, cv_, 0.1))

        # --- set-piece configurations ---
        # kick-off: home all on one side of halfway, away all on the other
        sep = np.fmax(fl_t[0] * (1 - fl_t[1]), (1 - fl_t[0]) * fl_t[1])
        add("ko_split", sep)
        dc = np.sqrt((x - L / 2) ** 2 + (y - W / 2) ** 2)
        add("n_circle", (dc < 9.15).sum(1))
        dcs = np.sort(dc, 1)
        add("d_spot1", np.where(cv_ > 0, dcs[:, 0], np.nan))
        add("d_spot2", np.where(cv_ > 1, dcs[:, 1], np.nan))
        # boxes (16.5 m deep, 40.32 m wide) at both ends; the busier one
        in_y = np.abs(y - W / 2) < 20.16
        boxL = (x < 16.5) & in_y
        boxR = (x > L - 16.5) & in_y
        nL, nR = boxL.sum(1), boxR.sum(1)
        busyL = nL >= nR
        box = np.where(busyL[:, None], boxL, boxR)
        add("n_box_busy", np.maximum(nL, nR))
        add("n_box_quiet", np.minimum(nL, nR))
        bt = [(box & tm[None, :]).sum(1) for tm in teams]
        pair("n_box_team", bt[0], bt[1])
        sixL = (x < 5.5) & (np.abs(y - W / 2) < 9.16)
        sixR = (x > L - 5.5) & (np.abs(y - W / 2) < 9.16)
        add("n_six", np.maximum(sixL.sum(1), sixR.sum(1)))
        # just outside the busier box (players waiting at the edge: penalties, goal kicks)
        edgeL = (x >= 16.5) & (x < 26.0) & in_y
        edgeR = (x <= L - 16.5) & (x > L - 26.0) & in_y
        add("n_box_edge", np.where(busyL, edgeL.sum(1), edgeR.sum(1)))
        # final third occupancy of the busier end
        add("n_third_busy", np.maximum((x < L / 3).sum(1), (x > 2 * L / 3).sum(1)))
        # corners / goal line / touchline
        dgl = np.fmin(x, L - x)
        dtl = np.fmin(y, W - y)
        dflag = np.sqrt(dgl ** 2 + dtl ** 2)
        add("n_flag5", (dflag < 5.0).sum(1))
        add("d_flag_min", np.where(cv_ > 0, np.nanmin(np.where(valid, dflag, np.inf), 1), np.nan))
        add("n_goalline2", (dgl < 2.0).sum(1))
        add("n_touch1", (dtl < 1.0).sum(1))
        add("d_touch_min", np.where(cv_ > 0, np.nanmin(np.where(valid, dtl, np.inf), 1), np.nan))
        add("d_gl_min", np.where(cv_ > 0, np.nanmin(np.where(valid, dgl, np.inf), 1), np.nan))
        add("n_outside", ((x < 0) | (x > L) | (y < 0) | (y > W)).sum(1))

    # --- pairwise: wall, marking, isolation (chunked) ---
    wall = np.zeros(n, np.float32)
    nn_opp = np.full(n, np.nan, np.float32)
    nn_mate = np.full(n, np.nan, np.float32)
    iso = np.full(n, np.nan, np.float32)
    same = slot_team[:, None] == slot_team[None, :]
    eye = np.eye(len(slot_team), dtype=bool)
    for a in range(0, n, 8000):
        b = min(n, a + 8000)
        q = P[a:b]
        d = np.sqrt(((q[:, :, None, :] - q[:, None, :, :]) ** 2).sum(-1))
        d = np.where(np.isnan(d), np.inf, d)
        d[:, eye] = np.inf
        dm = np.where(same[None], d, np.inf)
        do = np.where(~same[None], d, np.inf)
        wall[a:b] = (dm < 1.5).sum(2).max(1)
        vq = ~np.isnan(q[:, :, 0])
        no = do.min(2)
        nm = dm.min(2)
        with np.errstate(invalid="ignore"):
            no = np.where(vq & np.isfinite(no), no, np.nan)
            nm = np.where(vq & np.isfinite(nm), nm, np.nan)
            nn_opp[a:b] = np.nanmedian(no, 1)
            nn_mate[a:b] = np.nanmedian(nm, 1)
            iso[a:b] = np.nanmax(np.fmin(no, nm), 1)
    add("wall", wall)
    add("nn_opp_med", nn_opp)
    add("nn_mate_med", nn_mate)
    add("isolation_max", iso)

    base = dict(zip(names, cols))
    assert len(names) == N_BASE
    # --- temporal context ---
    s1 = int(fps)
    smooth_core = {}
    ctx_keys = ["speed_mean", "speed_p90", "walk_share", "still_share", "acc_mean",
                "coherence_all", "n_tracked", "ko_split", "n_box_busy", "n_flag5", "n_touch1",
                "wall", "nn_opp_med", "length", "cx_all_half", "n_circle", "d_flag_min",
                "d_touch_min", "speed_team_min", "speed_team_max", "isolation_max", "n_six"]
    for kname in ctx_keys:
        val = base[kname].astype(np.float64)
        for w_s in (1.0, 4.0, 10.0, 30.0):
            hw = int(w_s * fps / 2)
            m = mmean(val, -hw, hw)
            if w_s == 1.0:
                smooth_core[kname] = m
            add("{}_c{:g}".format(kname, w_s), m)
        for w_s in (2.0, 8.0):
            wf = int(w_s * fps)
            past = mmean(val, -wf, -1)
            fut = mmean(val, 1, wf)
            add("{}_p{:g}".format(kname, w_s), past)
            add("{}_f{:g}".format(kname, w_s), fut)
            add("{}_d{:g}".format(kname, w_s), fut - past)
    ms1 = smooth_core["speed_mean"]
    w5 = int(5 * fps)
    add("speed_min_p5", _roll_ext(ms1, -w5, 0, "min"))
    add("speed_min_f5", _roll_ext(ms1, 0, w5, "min"))
    add("speed_max_p5", _roll_ext(ms1, -w5, 0, "max"))
    add("speed_max_f5", _roll_ext(ms1, 0, w5, "max"))
    w15 = int(15 * fps)
    add("speed_max_c15", _roll_ext(ms1, -w15, w15, "max"))
    add("speed_min_c15", _roll_ext(ms1, -w15, w15, "min"))
    for kname in ("speed_mean", "still_share", "acc_mean", "n_tracked", "length", "ko_split",
                  "n_box_busy", "wall"):
        sm = smooth_core[kname]
        for off in CORE_SHIFT_S:
            kk = int(round(off * fps))
            add("{}_s+{:g}".format(kname, off), shift(sm, kk))
            add("{}_s-{:g}".format(kname, off), shift(sm, -kk))
    del s1
    F = np.stack(cols, 1).astype(np.float32)
    return F, names
