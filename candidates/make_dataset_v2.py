#!/usr/bin/env python3
"""Training data on the realistic degradation: DFL games degraded with a randomised degrade_v2.

    python -I candidates/make_dataset_v2.py [--out $MTAH_WORK/ds_v2] [--jobs 4]

The same output format as learned_dfl_evidence/make_dataset.py, so every existing training script
reads it unchanged:

- <out>/ds/<match>_<seed>.npz: LDE features (features.compute) on a stride-5 subsample of graded
  frames, with labels y (0 H, 1 A, 2 D) and the hand-over target yc.
- <out>/raw/<match>_<seed>.npz: the degraded slots / slot_team / ball plus codes, blocks and
  sign, for models that train on positions.

Seeds 41-43 never coincide with the bench's (2027-2029). Each degradation draws every
degrade_v2 knob around the values fitted to the competition tracking (domain randomisation), so a
model cannot overfit one simulator setting. It also adds a small AR(1) ball offset (sd up to
0.5 m), since the real ball's systematic error is unknown.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LDE = os.path.join(HERE, "learned_dfl_evidence")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, LDE)
import degrade_v2 as dv          # noqa: E402
import features as fe            # noqa: E402
import make_dataset as lde_ds    # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SEEDS = [41, 42, 43]
STRIDE = 5
FPS = 25
PITCH = (105.0, 68.0)


def draw_params(rng):
    """degrade_v2 knobs, randomised around the fitted DEFAULTS."""
    return dict(lead_s=rng.uniform(0.5, 4.0), reveal_s=rng.uniform(0.0, 3.0),
                p_keep_dead=rng.uniform(0.1, 0.35), p_split=rng.uniform(0.15, 0.5),
                split_s=rng.uniform(0.2, 1.5), alive_gaps_per90=rng.uniform(10, 60),
                alive_gap_s=rng.uniform(0.3, 1.5), crowd_boost=rng.uniform(1.5, 4.0),
                tracklet_s=rng.uniform(40, 160), player_gap_s=rng.uniform(3, 12),
                slots_per_team=int(rng.integers(21, 25)), smooth_frames=int(rng.choice([7, 9, 11, 13, 15, 19])),
                ball_smooth_frames=int(rng.choice([1, 3, 5])), ball_offset_sd=rng.uniform(0.0, 0.5))


def degrade(g, seed, key):
    codes, blocks = lde_ds.labels_blocks(g)
    rng = np.random.default_rng([seed, key])
    p = draw_params(rng)
    graded = codes != 3
    slots, slot_team, ball = dv.degrade_v2(g["players"].astype(np.float64), g["team_of_player"].astype(int),
                                           g["ball"].astype(np.float64), g["status"], graded,
                                           np.random.default_rng([seed, key, 1]), p)
    n = len(ball)
    a = np.exp(-1.0 / FPS)                                   # AR(1) offset, tau 1 s
    eps = rng.normal(0, 1, (n, 2)) * p["ball_offset_sd"] * np.sqrt(1 - a * a)
    off = np.zeros((n, 2))
    for t in range(1, n):
        off[t] = a * off[t - 1] + eps[t]
    ball = (ball + off).astype(np.float32)
    sign = lde_ds.attack_sign(g, blocks)
    return slots, slot_team, ball, codes, blocks, sign, p


def job(args):
    path, seed, out = args
    match = os.path.basename(path)[:-4]
    ds_path = os.path.join(out, "ds", "{}_{}.npz".format(match, seed))
    raw_path = os.path.join(out, "raw", "{}_{}.npz".format(match, seed))
    if os.path.isfile(ds_path) and os.path.isfile(raw_path):
        return "{} seed {}: exists".format(match, seed)
    t0 = time.time()
    g = lde_ds.load_game(path)
    slots, slot_team, ball, codes, blocks, sign, p = degrade(g, seed, lde_ds.game_key(match))
    fs = fe.frame_sign(len(ball), sign, blocks)
    F, names, kinds = fe.compute(slots, slot_team, ball, fs, PITCH, FPS)
    graded = np.zeros(len(codes), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    graded &= codes != 3
    idx = np.where(graded)[0]
    idx = idx[(idx % STRIDE) == (seed % STRIDE)]
    np.savez_compressed(ds_path, X=F[idx], y=codes[idx], yc=lde_ds.change_target(codes)[idx], idx=idx,
                        names=np.array(names), kinds=np.array(kinds))
    np.savez_compressed(raw_path, slots=slots, slot_team=slot_team, ball=ball, codes=codes,
                        blocks=np.array(blocks), sign=sign)
    with open(raw_path[:-4] + ".json", "w") as handle:
        json.dump({k: (float(v) if isinstance(v, float) else v) for k, v in p.items()}, handle)
    return "{} seed {}: {} rows, ball missing {:.2f}, {:.0f}s".format(
        match, seed, len(idx), float(np.isnan(ball[graded, 0]).mean()), time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "ds_v2"))
    ap.add_argument("--npz", default=os.path.join(WORK, "idsse", "npz"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    for sub in ("ds", "raw"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as handle:
        json.dump(dict(seeds=SEEDS, stride=STRIDE, degrade="degrade_v2 randomised (draw_params)",
                       defaults=dv.DEFAULTS), handle, indent=1)
    paths = sorted(os.path.join(args.npz, f) for f in os.listdir(args.npz) if f.endswith(".npz"))
    jobs = [(p, s, args.out) for p in paths for s in SEEDS]
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for msg in pool.imap_unordered(job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
