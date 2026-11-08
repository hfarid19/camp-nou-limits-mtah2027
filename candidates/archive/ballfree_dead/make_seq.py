#!/usr/bin/env python3
"""Sequence dataset for the ball-free TCN: per-frame base features_bf columns at 5 Hz.

    python -I candidates/ballfree_dead/make_seq.py --npz <idsse npz dir> --out <dir> [--jobs 2]

Same degradations as make_dataset.py (LDE's degrade_dfl, seeds 41-43). Stored per game and seed:
X (T, K) = the BASE (non-temporal) features_bf columns at frames 0, 5, 10, ..., y (T,) the
tracking labels there (0 H, 1 A, 2 D, 3 ungraded). The TCN learns its own temporal context.
"""
import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_dataset as md     # noqa: E402  (this folder's: degrade via LDE + features_bf)
import features_bf as fb      # noqa: E402

STEP = 5


def job(args):
    npz, out_dir, match, seed = args
    out = os.path.join(out_dir, "{}_{}.npz".format(match, seed))
    if os.path.isfile(out):
        return
    t0 = time.time()
    g = md.lde_md.load_game(os.path.join(npz, match + ".npz"))
    slots, slot_team, ball, codes, blocks, sign, info = md.lde_md.degrade_dfl(
        g, seed, md.lde_md.game_key(match))
    F, names = fb.compute(slots, slot_team, md.PITCH, md.FPS)
    graded = np.zeros(len(codes), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    y = np.where(graded, codes, 3).astype(np.int8)
    k = fb.N_BASE
    np.savez_compressed(out, X=F[::STEP, :k], y=y[::STEP], names=np.array(names[:k]))
    print("{} seed {}: {} steps, {:.0f}s".format(match, seed, len(y[::STEP]), time.time() - t0),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    matches = sorted(f[:-4] for f in os.listdir(a.npz) if f.endswith(".npz"))
    jobs = [(a.npz, a.out, m, s) for m in matches for s in md.SEEDS]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        list(pool.imap_unordered(job, jobs))


if __name__ == "__main__":
    main()
