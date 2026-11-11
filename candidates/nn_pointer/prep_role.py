#!/usr/bin/env python3
"""Role-aware pointer inputs (role.py, r5_ptr_role) for every ds file (run where the ds raw files are):

    python -I candidates/nn_pointer/prep_role.py --ds $MTAH_WORK/ds_pff2 [--jobs 16]

<ds>/ptrr/<name>_<seed>.npz: xs (T, 8, 22) float16, ms (T, 8) uint8, xf (T, 26) float16, sel (T, 8) int16, idx =
role.prepare on the file's degraded slots / ball at the ds file's idx rows (same call as prep_ptr.py --ballc, which
role.prepare extends: the first 10 / 14 columns are ballc's).
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
import role                # noqa: E402


def build(ds_path, out_path):
    if os.path.isfile(out_path):
        return "exists " + out_path
    idx = np.load(ds_path)["idx"].astype(np.int64)
    raw = np.load(os.path.join(os.path.dirname(os.path.dirname(ds_path)), "raw", os.path.basename(ds_path)))
    sf = rc.frame_sign(len(raw["ball"]), float(raw["sign"]), [tuple(int(v) for v in b) for b in raw["blocks"]])
    inp = role.prepare(raw["slots"], raw["slot_team"], raw["ball"], sf, 25.0, idx)
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, xs=inp["xs"].astype(np.float16), ms=inp["ms"].astype(np.uint8), xf=inp["xf"].astype(np.float16),
             sel=inp["sel"].astype(np.int16), idx=idx)
    os.replace(tmp, out_path)
    return "{} T {}".format(os.path.basename(out_path), len(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default=None, help="substring filter on file names")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.ds, "ptrr"), exist_ok=True)
    files = [p for p in sorted(glob.glob(os.path.join(a.ds, "ds", "*_*.npz"))) if not a.only or a.only in p]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for m in pool.starmap(build, [(p, os.path.join(a.ds, "ptrr", os.path.basename(p))) for p in files]):
            print(m, flush=True)


if __name__ == "__main__":
    main()
