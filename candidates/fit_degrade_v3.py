#!/usr/bin/env python3
"""Fit degrade_v3's player knobs to BAR-ATH's label-free tracking statistics.

    python candidates/fit_degrade_v3.py [--evals 60]

Ball knobs are degrade_v2's (already fitted). Only the player model changes, so only player
statistics are matched: coverage, tracklet count and length quantiles, and the share of tracklets
ending in a crowd. Positions of the match are read, never a label.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bench                  # noqa: E402
import degrade_v3 as d3       # noqa: E402
import fit_degrade_v2 as f2   # noqa: E402
import match_stats as ms      # noqa: E402

PLAYER_KEYS = {"visible_home": 3.0, "visible_away": 3.0, "tracklets_per90": 1.0,
               "tracklet_p10_s": 1.0, "tracklet_p50_s": 1.5, "tracklet_p90_s": 1.0,
               "tracklet_mean_s": 0.5, "tracklet_end_crowd": 2.0}
PLAYER_SPACE = dict(occ_r=(0.5, 3.0), p_occ=(0.0, 1.0), occ_gap_s=(0.2, 10.0),
                    tracklet_s=(60.0, 900.0), player_gap_s=(0.5, 20.0))


def player_stats(g, p, seed):
    rng = np.random.default_rng(seed)
    slots, slot_team = d3.degrade_players(g["players"], g["team"], rng, p)
    return ms.stats(slots, slot_team, g["ball"], g["blocks"], d3.FPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=60)
    args = ap.parse_args()
    want = ms.stats(*ms.match_input())
    names = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    fit_games = f2.load_dfl(names[:2])
    rng = np.random.default_rng(2027)
    p, d = f2.search(fit_games, player_stats, PLAYER_SPACE, PLAYER_KEYS, want, args.evals, rng,
                     dict(d3.DEFAULTS))
    print("FITTED", {k: round(float(p[k]), 3) for k in PLAYER_SPACE}, "distance {:.3f}".format(d))
    rows = []
    for i, g in enumerate(f2.load_dfl(names)):
        slots, slot_team, ball = d3.degrade_v3(g["players"], g["team"], g["ball"], g["status"],
                                               g["graded"], np.random.default_rng(100 + i), p)
        rows.append(ms.stats(slots, slot_team, ball, g["blocks"], d3.FPS))
    ms.table({"BAR-ATH": want, "DFL v3 (7)": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}})


if __name__ == "__main__":
    main()
