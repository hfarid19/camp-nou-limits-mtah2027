#!/usr/bin/env python3
"""handover_mstcn_v2: copy of handover_mstcn/prep_data.py plus --target raw|ctrl. With ctrl the
full-resolution codes are relabelled with control.fold(codes, --ctrl-s, valid_runs) BEFORE the
class targets and both boundary targets are derived (see README.md).

Turn the domain-randomised DFL feature datasets into MS-TCN training sequences.

    python -I prep_data.py --ds IDSSE_R2/ds --npz IDSSE/npz --out OUT

Input: learned_dfl_evidence/make_dataset.py output (round-2 settings, degradation seeds 11/12/13,
features.py columns at every graded frame t with t % 5 == seed % 5, i.e. one 5 fps phase per seed)
and the parsed raw DFL games (per-frame BallStatus / BallPossession flags).
Output, per game and seed: OUT/<match>_<seed>.npz with
    X     (T, K) float32 features (NaN kept)
    y     (T,) int8 0 H / 1 A / 2 D / 3 ungraded
    bha   (T,) float32 soft H<->A boundary target  exp(-d^2 / 2 sigma^2), d = frames to the
          nearest direct H<->A flag change (change located half-way between the two frames)
    bd    (T,) float32 same for alive<->dead changes
    seq   (T,) int16 sequence id (one per half)
    idx   (T,) frame index
Never touches the bench's own DFL caches (bench_cache/dfl_*).
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "evaluation"))
sys.path.insert(0, os.path.dirname(HERE))                # candidates/ (control)
import control  # noqa: E402
import possession_metrics as pm  # noqa: E402

SIGMA = 2.5   # frames (25 fps)


def labels(g):
    st, po = g["status"], g["poss"]
    codes = np.full(len(st), 3, np.int8)
    codes[(st == 1) & (po == 1)] = 0
    codes[(st == 1) & (po == 2)] = 1
    codes[st == 0] = 2
    return codes


def change_positions(codes):
    a, b = codes[:-1], codes[1:]
    ha = np.where((a != b) & (a < 2) & (b < 2))[0] + 0.5
    dd = np.where((a != b) & (((a == 2) & (b < 2)) | ((a < 2) & (b == 2))))[0] + 0.5
    return ha, dd


def soft_target(frames, pos, sigma=SIGMA):
    if len(pos) == 0:
        return np.zeros(len(frames), np.float32)
    j = np.clip(np.searchsorted(pos, frames), 1, len(pos) - 1)
    d = np.minimum(np.abs(frames - pos[j - 1]), np.abs(frames - pos[j]))
    if len(pos) == 1:
        d = np.abs(frames - pos[0])
    return np.exp(-d ** 2 / (2 * sigma ** 2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["raw", "ctrl"], default="raw")
    ap.add_argument("--ctrl-s", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.ds, "DFL-MAT-*_*.npz"))):
        match, seed = os.path.basename(p)[:-4].rsplit("_", 1)
        z = np.load(p)
        g = np.load(os.path.join(a.npz, match + ".npz"))
        codes = labels(g)
        idx = z["idx"].astype(np.int64)
        rp = os.path.join(os.path.dirname(os.path.abspath(a.ds)), "raw", os.path.basename(p))
        if os.path.isfile(rp):                   # make_dataset_v2: full-resolution codes saved
            rc = np.load(rp)["codes"].astype(np.int8)
            assert np.array_equal(rc, codes), rp
        assert np.array_equal(codes[idx], z["y"]), p
        if a.target == "ctrl":                   # controlled possession (bench.controlled_labels)
            codes = control.fold(codes, a.ctrl_s, pm.valid_runs(codes), 25)
        ha, dd = change_positions(codes)
        half = g["half"][idx]
        # sequence breaks: half change or a hole in the stride-5 grid
        brk = np.concatenate(([True], (np.diff(idx) != 5) | (np.diff(half) != 0)))
        seq = np.cumsum(brk).astype(np.int16) - 1
        fr = idx.astype(np.float64)
        np.savez(os.path.join(a.out, "{}_{}.npz".format(match, seed)), X=z["X"], y=codes[idx],
                 bha=soft_target(fr, ha), bd=soft_target(fr, dd), seq=seq, idx=idx,
                 names=z["names"], kinds=z["kinds"])
        print(match, seed, len(idx), "seqs", seq.max() + 1, "ha", len(ha), "dd", len(dd),
              "y==z.y", bool((codes[idx] == z["y"]).all()), a.target, flush=True)


if __name__ == "__main__":
    main()
