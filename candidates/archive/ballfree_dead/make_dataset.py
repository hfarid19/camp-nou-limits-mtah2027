#!/usr/bin/env python3
"""Ball-free training set: degraded DFL/IDSSE games -> features_bf rows + dead/alive labels.

    python -I candidates/ballfree_dead/make_dataset.py --npz <idsse npz dir> --out <dir> [--jobs 2]

Each of the 7 open DFL games (sorted file names = the bench's game index) is degraded with
learned_dfl_evidence/make_dataset.degrade_dfl (the bench's tune_on_sample_data.degrade with
domain randomisation: missing ball 33-48 %, tracklet length 30-100 s, ghosts, far-side dropouts,
stoppage blackouts with tracked-player collapse, low-passed ball ...) with SEEDS below, which are
neither the bench's seed (2027) nor LDE's (11-13). The ball is degraded too but never used here:
only the player slots enter features_bf. Labels: 1 where BallStatus == 0 (dead), 0 alive;
rows: graded frames with stride STRIDE (offset per seed). -> <out>/<match>_<seed>.npz
"""
import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LDE = os.path.join(ROOT, "candidates", "learned_dfl_evidence")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, LDE)              # LDE's make_dataset imports its own features module
sys.path.insert(0, HERE)
import importlib.util                # noqa: E402
_spec = importlib.util.spec_from_file_location("lde_make_dataset", os.path.join(LDE, "make_dataset.py"))
lde_md = importlib.util.module_from_spec(_spec)     # LDE's degradation, loaded unchanged
_spec.loader.exec_module(lde_md)
import features_bf as fb             # noqa: E402

SEEDS = [41, 42, 43]
STRIDE = 8
PITCH = (105.0, 68.0)
FPS = 25


def job(args):
    npz, out_dir, match, seed = args
    out = os.path.join(out_dir, "{}_{}.npz".format(match, seed))
    if os.path.isfile(out):
        return out
    t0 = time.time()
    g = lde_md.load_game(os.path.join(npz, match + ".npz"))
    gi = lde_md.game_key(match)
    slots, slot_team, ball, codes, blocks, sign, info = lde_md.degrade_dfl(g, seed, gi)
    F, names = fb.compute(slots, slot_team, PITCH, FPS)
    graded = np.zeros(len(codes), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    graded &= codes != 3
    idx = np.where(graded)[0]
    idx = idx[(idx % STRIDE) == (seed % STRIDE)]
    np.savez_compressed(out, X=F[idx], y=codes[idx], idx=idx, names=np.array(names))
    print("{} seed {}: {} rows, {} features, blackout {}, {:.0f}s".format(
        match, seed, len(idx), F.shape[1], int(info["blackout"]), time.time() - t0), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    matches = sorted(f[:-4] for f in os.listdir(a.npz) if f.endswith(".npz"))
    jobs = [(a.npz, a.out, m, s) for m in matches for s in SEEDS]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        list(pool.imap_unordered(job, jobs))


if __name__ == "__main__":
    main()
