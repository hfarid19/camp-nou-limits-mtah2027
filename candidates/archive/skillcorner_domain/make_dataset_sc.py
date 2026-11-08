#!/usr/bin/env python3
"""Degrade the parsed SkillCorner games and compute the LDE training features.

    python -I make_dataset_sc.py --npz DIR --out DIR [--seeds 21] [--players ex|det] [--jobs 2]

SkillCorner broadcast tracking is already "automatic-like": the ball is detected on ~60% of
graded frames (~93% while alive, ~20% in stoppages: replays and close-ups), which matches the
competition's 41% missing ball concentrated in stoppages without any synthetic blackout. On top
of it, per game and seed:
  1. players: "ex" (default) uses SkillCorner's extrapolated positions for all 22 players (the
     competition tracks ~20.5 objects during play; detected-only broadcast gives ~9), "det" keeps
     only players detected on the broadcast image.
  2. tune_on_sample_data.degrade: anonymous fragmented slots (mean tracklet ~ U(30, 100) s),
     short slot dropouts, N(0, 0.2 m) ball noise; target_missing ~ U(0.33, 0.48) only adds
     crowd-dependent ball gaps when the broadcast ball is seen more often than that.
  3. ghost duplicate slots on ~2% of frames (as make_dataset.degrade_dfl).
  4. features.compute (identical to LDE), labels from parse_skillcorner (D where the possession
     group is null, else H/A), stride-5 subsample of graded frames, change target +-4 frames.
-> <out>/SC-<id>_<seed>.npz with X, y, yc, idx, names, kinds (same layout as LDE's ds files).
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import tune_on_sample_data as tsd   # noqa: E402
import features as fe               # noqa: E402
import make_dataset as md           # noqa: E402  (LDE's, copied: labels_blocks, change_target, ...)

warnings.filterwarnings("ignore")
STRIDE = 5
PITCH = (105.0, 68.0)
FPS = 25
ARGS = {}


def degrade_sc(g, seed, key, players_mode):
    codes, blocks = md.labels_blocks(g)
    codes[g["half"] == 0] = 3                   # outside the match periods: never graded
    rng = np.random.default_rng([seed, key])
    target = float(rng.uniform(0.33, 0.48))
    tracklet = float(rng.uniform(30, 100))
    pl = g["players_ex"] if players_mode == "ex" else g["players"]
    game = tsd.Game("sc", pl.astype(np.float64), g["team_of_player"].astype(int),
                    g["ball"].astype(np.float64), [])
    tsd.SEED = seed * 100 + key % 100
    graded = codes != 3
    slots, slot_team, ball = tsd.degrade(game, graded, np.random.default_rng([seed, key, 1]),
                                         target_missing=target, mean_tracklet_s=tracklet)
    n = len(ball)
    ghost = md._runs_mask(n, 0.02 / 25, 25, rng)
    for s, e in zip(*fe.runs_of(ghost)):
        cand = np.where(~np.isnan(slots[s, :, 0]))[0]
        free = np.where(np.isnan(slots[s:e, :, 0]).all(0))[0]
        if not len(cand) or not len(free):
            continue
        j = int(rng.choice(cand))
        free = free[slot_team[free] == slot_team[j]]
        if not len(free):
            continue
        k = int(rng.choice(free))
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.5, 1.0)
        slots[s:e, k] = slots[s:e, j] + np.array([r * np.cos(ang), r * np.sin(ang)], np.float32)
    sign = md.attack_sign(dict(players=pl, team_of_player=g["team_of_player"]), blocks)
    return slots, slot_team, ball.astype(np.float32), codes, blocks, sign, dict(target=target,
                                                                               tracklet=tracklet)


def one(job):
    path, seed = job
    match = os.path.basename(path)[:-4]
    out = os.path.join(ARGS["out"], "{}_{}.npz".format(match, seed))
    if os.path.isfile(out):
        return
    t0 = time.time()
    g = md.load_game(path)
    slots, slot_team, ball, codes, blocks, sign, info = degrade_sc(g, seed, zlib.crc32(match.encode()) % 100000,
                                                                   ARGS["players"])
    fs = fe.frame_sign(len(ball), sign, blocks)
    F, names, kinds = fe.compute(slots, slot_team, ball, fs, PITCH, FPS)
    graded = np.zeros(len(codes), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    graded &= codes != 3
    idx = np.where(graded)[0]
    idx = idx[(idx % STRIDE) == (seed % STRIDE)]
    yc = md.change_target(codes)
    np.savez_compressed(out, X=F[idx], y=codes[idx], yc=yc[idx], idx=idx, names=np.array(names),
                        kinds=np.array(kinds))
    print("{} seed {}: {} rows, ball missing {:.2f}, tracked {:.1f}, {:.0f}s".format(
        match, seed, len(idx), np.isnan(ball[graded, 0]).mean(),
        (~np.isnan(slots[graded, :, 0])).sum(1).mean(), time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="21")
    ap.add_argument("--players", default="ex", choices=["ex", "det"])
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ARGS.update(out=args.out, players=args.players)
    config = dict(seeds=args.seeds, stride=STRIDE, players=args.players)
    cpath = os.path.join(args.out, "config.json")
    if os.path.isfile(cpath):
        assert json.load(open(cpath)) == config, "{} was built with other settings".format(args.out)
    json.dump(config, open(cpath, "w"), indent=1)
    jobs = [(p, int(s)) for p in sorted(glob.glob(os.path.join(args.npz, "SC-*.npz")))
            for s in args.seeds.split(",")]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        pool.map(one, jobs, chunksize=1)


if __name__ == "__main__":
    main()
