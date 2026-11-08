#!/usr/bin/env python3
"""Fit degrade_v2's knobs so that degraded DFL tracking has BAR-ATH's label-free statistics.

    python candidates/fit_degrade_v2.py [--evals 120]

The targets are match_stats.stats() of the competition match's tracking (positions only, no
labels). Ball knobs and player knobs act on disjoint statistics, so they are fitted separately by
random search on 3 DFL games, then checked on all 7. Prints the fitted dict for
degrade_v2.DEFAULTS and the full comparison table.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bench                 # noqa: E402
import degrade_v2 as dv      # noqa: E402
import match_stats as ms     # noqa: E402

BALL_KEYS = {"ball_missing": 3.0, "gaps_per90": 1.0, "gap_p10_s": 0.5, "gap_p50_s": 1.0,
             "gap_p90_s": 1.0, "miss_in_gaps_gt5s": 2.0}
PLAYER_KEYS = {"visible_home": 3.0, "visible_away": 3.0, "tracklets_per90": 1.0,
               "tracklet_p50_s": 1.0, "tracklet_mean_s": 1.0, "player_accel_med": 1.0,
               "slots_used": 0.5}
BALL_SPACE = dict(lead_s=(0.0, 3.0), reveal_s=(0.0, 12.0), p_keep_dead=(0.0, 0.4),
                  p_split=(0.0, 0.8), split_s=(0.3, 4.0), alive_gaps_per90=(0.0, 80.0),
                  alive_gap_s=(0.2, 2.0))
PLAYER_SPACE = dict(tracklet_s=(30.0, 250.0), player_gap_s=(1.0, 20.0), smooth_frames=(1, 25),
                    slots_per_team=(21, 25))


def load_dfl(names):
    games = []
    for m in names:
        z = np.load(os.path.join(bench.DFL_DIR, m + ".npz"))
        half = z["half"]
        blocks = []
        for h in (1, 2):
            idx = np.where(half == h)[0]
            if len(idx):
                blocks.append((h, int(idx[0]), int(idx[-1])))
        graded = ms._graded(len(half), blocks)
        games.append(dict(players=z["players"].astype(np.float64), team=z["team_of_player"].astype(int),
                          ball=z["ball"].astype(np.float64), status=z["status"], graded=graded,
                          blocks=blocks))
    return games


def dist(got, want, keys):
    return sum(w * abs(got[k] - want[k]) / max(abs(want[k]), 1e-6) for k, w in keys.items())


def ball_stats(g, p, seed):
    rng = np.random.default_rng(seed)
    ball = g["ball"].copy()
    ball[dv.hide_ball(g["ball"], g["players"], g["status"], g["graded"], rng, p)] = np.nan
    return ms.stats(g["players"][:, :2], np.array([0, 1]), ball, g["blocks"], dv.FPS)


def player_stats(g, p, seed):
    rng = np.random.default_rng(seed)
    slots, slot_team = dv.degrade_players(g["players"], g["team"], rng, p)
    return ms.stats(slots, slot_team, g["ball"], g["blocks"], dv.FPS)


def mean_stats(games, fn, p, seed=7):
    rows = [fn(g, p, seed + i) for i, g in enumerate(games)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def search(games, fn, space, keys, want, evals, rng, base):
    best_p, best_d = dict(base), dist(mean_stats(games, fn, base), want, keys)
    for i in range(evals):
        p = dict(best_p)
        scale = 1.0 if i < evals // 2 else 0.3          # global, then local around the best
        for k, (lo, hi) in space.items():
            if isinstance(lo, int):
                v = int(round(best_p[k] + rng.normal(0, scale * (hi - lo) / 2)))
            else:
                v = best_p[k] + rng.normal(0, scale * (hi - lo) / 2)
            p[k] = type(lo)(min(hi, max(lo, v)))
        d = dist(mean_stats(games, fn, p), want, keys)
        if d < best_d:
            best_p, best_d = p, d
            print("  eval {:3d}  dist {:.3f}  {}".format(i, d, {k: round(float(p[k]), 2) for k in space}),
                  flush=True)
    return best_p, best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=120)
    args = ap.parse_args()
    want = ms.stats(*ms.match_input())
    names = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    fit_games = load_dfl(names[:3])
    rng = np.random.default_rng(2026)
    p = dict(dv.DEFAULTS)
    print("ball knobs")
    p, d_ball = search(fit_games, ball_stats, BALL_SPACE, BALL_KEYS, want, args.evals, rng, p)
    print("player knobs")
    p, d_play = search(fit_games, player_stats, PLAYER_SPACE, PLAYER_KEYS, want, args.evals // 3, rng, p)
    print("FITTED", {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in p.items()})
    print("distance ball {:.3f}  players {:.3f}".format(d_ball, d_play))

    all_games = load_dfl(names)
    rows = []
    for i, g in enumerate(all_games):
        slots, slot_team, ball = dv.degrade_v2(g["players"], g["team"], g["ball"], g["status"],
                                                g["graded"], np.random.default_rng(100 + i), p)
        rows.append(ms.stats(slots, slot_team, ball, g["blocks"], dv.FPS))
    got = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    old = []
    for m in names:
        slots, slot_team, ball, _, blocks, _ = bench.dfl_prepared(m, 2027)
        old.append(ms.stats(slots, slot_team, ball, blocks, dv.FPS))
    ms.table({"BAR-ATH": want, "DFL v2 (7)": got,
              "DFL v1 bench": {k: float(np.mean([r[k] for r in old])) for k in old[0]}})


if __name__ == "__main__":
    main()
