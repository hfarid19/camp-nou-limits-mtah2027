#!/usr/bin/env python3
"""Slot inputs of the possessor-pointer model for every ds file (run where the ds raw files are):

    python -I candidates/nn_pointer/prep_ptr.py --ds $MTAH_WORK/ds_pff2 [--jobs 16]

<ds>/ptr/<name>_<seed>.npz: xs (T, S, 11) float16, ms (T, S) uint8, xf (T, 10) float16 at the ds file's idx
rows = ball_radar_pff/radar_common.prepare on the file's degraded slots / ball (the same function and
arguments the pipeline uses at inference: full-length arrays, the frames of one stride-5 phase).
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.append(os.path.join(CAND, "ball_radar_pff"))
import radar_common as rc  # noqa: E402
sys.path.insert(0, HERE)
import ballc               # noqa: E402


def build_ballc(ds_path, out_path):
    """ball-centred inputs (ballc.prepare, K = 8) -> <ds>/ptrb/<file>: xs, ms, xf (float16 / uint8), sel, idx."""
    if os.path.isfile(out_path):
        return "exists " + out_path
    idx = np.load(ds_path)["idx"].astype(np.int64)
    raw = np.load(os.path.join(os.path.dirname(os.path.dirname(ds_path)), "raw", os.path.basename(ds_path)))
    sf = rc.frame_sign(len(raw["ball"]), float(raw["sign"]), [tuple(int(v) for v in b) for b in raw["blocks"]])
    inp = ballc.prepare(raw["slots"], raw["slot_team"], raw["ball"], sf, 25.0, idx)
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, xs=inp["xs"].astype(np.float16), ms=inp["ms"].astype(np.uint8), xf=inp["xf"].astype(np.float16),
             sel=inp["sel"].astype(np.int16), idx=idx)
    os.replace(tmp, out_path)
    return "{} T {}".format(os.path.basename(out_path), len(idx))


def build(ds_path, out_path):
    if os.path.isfile(out_path):
        return "exists " + out_path
    idx = np.load(ds_path)["idx"].astype(np.int64)
    raw = np.load(os.path.join(os.path.dirname(os.path.dirname(ds_path)), "raw", os.path.basename(ds_path)))
    slots, st, ball = raw["slots"], raw["slot_team"], raw["ball"]
    sf = rc.frame_sign(len(ball), float(raw["sign"]), [tuple(int(v) for v in b) for b in raw["blocks"]])
    inp = rc.prepare(slots, st, ball, sf, 25.0, idx)
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, xs=inp["xs"].astype(np.float16), ms=inp["ms"].astype(np.uint8), xf=inp["xf"].astype(np.float16), idx=idx)
    os.replace(tmp, out_path)
    return "{} T {} S {}".format(os.path.basename(out_path), len(idx), slots.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default=None, help="substring filter on file names")
    ap.add_argument("--ballc", action="store_true", help="ball-centred inputs (ballc.py) into <ds>/ptrb")
    a = ap.parse_args()
    sub, fn = ("ptrb", build_ballc) if a.ballc else ("ptr", build)
    os.makedirs(os.path.join(a.ds, sub), exist_ok=True)
    files = [p for p in sorted(glob.glob(os.path.join(a.ds, "ds", "*_*.npz"))) if not a.only or a.only in p]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for m in pool.starmap(fn, [(p, os.path.join(a.ds, sub, os.path.basename(p))) for p in files]):
            print(m, flush=True)


if __name__ == "__main__":
    main()
