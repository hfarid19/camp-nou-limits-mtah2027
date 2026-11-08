#!/usr/bin/env python3
"""Label-free statistics of a tracking feed, to check that the bench's simulated degradation looks
like the competition's real automatic tracking.

    python candidates/match_stats.py              # table: BAR-ATH vs degraded DFL / Metrica

Everything here is computed from positions alone (players, ball, graded half blocks); no label of
any game is read, so running it on the competition match is inspecting the data, not tuning on
answers (RULES.md). The same function scores the simulator's output, so the knobs of
degrade_v2.py can be fitted by matching these numbers.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import possession_pipeline as pp     # noqa: E402

CROWD_M, CROWD_N = 3.0, 3            # "ball in a crowd": >= 3 tracked players within 3 m


def _graded(n, blocks):
    g = np.zeros(n, bool)
    for _, first, last in blocks:
        g[first:last + 1] = True
    return g


def _runs(mask):
    s, e = pp.runs_of(mask)
    return np.asarray(s), np.asarray(e)


def _jitter(xy, fps, min_run=11):
    """Median distance (m) of a track from its own 5-frame centred mean, over visible runs."""
    seen = ~np.isnan(xy[:, 0])
    out = []
    for s, e in zip(*_runs(seen)):
        if e - s < min_run:
            continue
        seg = xy[s:e].astype(np.float64)
        k = np.ones(5) / 5
        sm = np.stack([np.convolve(seg[:, j], k, mode="valid") for j in range(2)], 1)
        out.append(np.linalg.norm(seg[2:-2] - sm, axis=1))
    return float(np.median(np.concatenate(out))) if out else float("nan")


def _accel(xy, fps):
    """Median |second difference| of a track in m/s^2 over frames with 3 consecutive observations."""
    a = (xy[2:] - 2 * xy[1:-1] + xy[:-2]) * fps * fps
    a = np.linalg.norm(a, axis=1)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if len(a) else float("nan")


def stats(players, slot_team, ball, blocks, fps):
    n = len(ball)
    g = _graded(n, blocks)
    mins = g.sum() / fps / 60.0
    per90 = 90.0 / mins
    miss = np.isnan(ball[:, 0])
    out = {"graded_min": round(mins, 1), "ball_missing": float(miss[g].mean())}

    # ball gaps inside the graded halves
    s, e = _runs(miss & g)
    L = (e - s) / fps
    out["gaps_per90"] = len(L) * per90
    for q in (10, 50, 90):
        out["gap_p%d_s" % q] = float(np.percentile(L, q)) if len(L) else 0.0
    out["gap_mean_s"] = float(L.mean()) if len(L) else 0.0
    out["miss_in_gaps_gt5s"] = float(L[L > 5].sum() / L.sum()) if len(L) else 0.0

    # do gaps start more often when the last seen ball is in a crowd?
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    crowd = (np.nan_to_num(d, nan=99.0) < CROWD_M).sum(1) >= CROWD_N
    seen = ~miss & g
    starts = np.zeros(n, bool)
    starts[s[s > 0] - 1] = True                    # last visible frame before each gap
    starts &= seen
    pc = starts[seen & crowd].mean() if (seen & crowd).any() else np.nan
    po = starts[seen & ~crowd].mean() if (seen & ~crowd).any() else np.nan
    out["crowd_share"] = float(crowd[seen].mean())
    out["gap_start_crowd_ratio"] = float(pc / po) if po else float("nan")

    # ball smoothness and noise
    bg = np.where(g[:, None], ball, np.nan)
    out["ball_jitter_m"] = _jitter(bg, fps)
    out["ball_accel_med"] = _accel(bg, fps)
    v = np.linalg.norm(np.diff(bg, axis=0), axis=1) * fps
    v = v[~np.isnan(v)]
    out["ball_speed_p50"] = float(np.median(v))
    out["ball_speed_p90"] = float(np.percentile(v, 90))

    # player tracks: slots, tracklets, coverage, jitter
    P = players[g]
    vis = ~np.isnan(P[:, :, 0])
    for t, name in ((0, "home"), (1, "away")):
        out["visible_%s" % name] = float(vis[:, slot_team == t].sum(1).mean())
    out["slots_used"] = int(vis.any(0).sum())
    lens, ends = [], []
    for k in range(players.shape[1]):
        ks, ke = _runs(~np.isnan(players[:, k, 0]) & g)
        lens.extend((ke - ks) / fps)
        ends.extend((k, e - 1) for e in ke if e < len(g) and g[e])   # ended inside graded play
    lens = np.asarray(lens)
    out["tracklets_per90"] = len(lens) * per90
    out["tracklet_p10_s"] = float(np.percentile(lens, 10)) if len(lens) else 0.0
    out["tracklet_p50_s"] = float(np.median(lens)) if len(lens) else 0.0
    out["tracklet_p90_s"] = float(np.percentile(lens, 90)) if len(lens) else 0.0
    out["tracklet_mean_s"] = float(lens.mean()) if len(lens) else 0.0
    # share of tracklet ends with another tracked player within 2 m (occlusion-driven breaks)
    crowded = []
    for k, t in ends[::max(1, len(ends) // 4000)]:
        d = np.linalg.norm(players[t] - players[t, k], axis=1)
        d[k] = np.inf
        crowded.append(bool(np.nanmin(np.where(np.isnan(d), np.inf, d)) < 2.0))
    out["tracklet_end_crowd"] = float(np.mean(crowded)) if crowded else 0.0
    sub = P[::5]
    spd = np.linalg.norm(np.diff(sub, axis=0), axis=2) * fps / 5
    out["stationary_share"] = float(np.nanmean(np.where(np.isnan(spd), np.nan, spd < 0.7)))
    jit = [_jitter(players[:, k], fps) for k in range(0, players.shape[1], 3)]
    out["player_jitter_m"] = float(np.nanmedian(jit))
    acc = [_accel(np.where(g[:, None], players[:, k], np.nan), fps) for k in range(0, players.shape[1], 3)]
    out["player_accel_med"] = float(np.nanmedian(acc))
    return out


def match_input():
    """The competition match's tracking, exactly as make_submission.py reads it."""
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    slot_team, _, _, _ = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    return players, slot_team, ball, half_video_blocks(GAME), GAMES[GAME]["fps"]


def table(rows):
    keys = list(next(iter(rows.values())).keys())
    names = list(rows)
    print("{:24s}".format("") + "".join("{:>14s}".format(n[:13]) for n in names))
    for k in keys:
        cells = []
        for n in names:
            v = rows[n][k]
            cells.append("{:>14.3f}".format(v) if isinstance(v, float) else "{:>14}".format(v))
        print("{:24s}".format(k) + "".join(cells))


def main():
    import bench
    rows = {"BAR-ATH": stats(*match_input())}
    matches = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    dfl = [stats(slots, slot_team, ball, blocks, bench.tsd.FPS)
           for slots, slot_team, ball, _, blocks, _ in (bench.dfl_prepared(m, 2027) for m in matches)]
    rows["DFL bench"] = {k: float(np.mean([r[k] for r in dfl])) for k in dfl[0]}
    met = [stats(slots, slot_team, ball, blocks, bench.tsd.FPS)
           for slots, slot_team, ball, _, blocks, _ in (bench.prepared(g, 2027) for g in bench.DEV)]
    rows["Metrica dev"] = {k: float(np.mean([r[k] for r in met])) for k in met[0]}
    table(rows)


if __name__ == "__main__":
    main()
