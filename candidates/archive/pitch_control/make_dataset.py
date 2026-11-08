#!/usr/bin/env python3
"""Pitch-control training features on LDE's domain-randomised DFL degradations.

    python -I make_dataset.py --variant r1|r2 --out DIR [--lde-ds DIR]

Regenerates exactly the degradations of learned_dfl_evidence/make_dataset.py (copied unchanged
as lde_make_dataset.py; seeds 11, 12, 13; r1 = --drop-p 0 --blackout-p 0 for the team model,
r2 = defaults for the dead / hand-over models), computes LDE's features (features.py, copied
unchanged) and the pitch-control features (pc.py) on every frame, and saves the stride-5 rows
LDE trained on: DIR/<match>_<seed>.npz with X (LDE 194 cols), P (pitch-control cols), y, yc, idx.
With --lde-ds, the regenerated LDE features are checked bit-for-bit against LDE's stored dataset
(proves the degradations are the same ones). Never touches the bench's DFL caches (seed 2027).
"""
import argparse, glob, os, sys, time, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lde_make_dataset as md   # noqa: E402
import features as fe           # noqa: E402
import pc                       # noqa: E402
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--variant", choices=["r1", "r2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lde-ds", default=None)
    a = ap.parse_args()
    if a.variant == "r1":
        md.DROP_P, md.BLACKOUT_P = 0.0, 0.0
    os.makedirs(a.out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(a.npz, "DFL-MAT-*.npz"))):
        match = os.path.basename(path)[:-4]
        g = md.load_game(path)
        gi = md.game_key(match)
        for seed in md.SEEDS:
            out = os.path.join(a.out, "{}_{}.npz".format(match, seed))
            if os.path.isfile(out):
                continue
            t0 = time.time()
            slots, slot_team, ball, codes, blocks, sign, info = md.degrade_dfl(g, seed, gi)
            fs = fe.frame_sign(len(ball), sign, blocks)
            F, names, kinds = fe.compute(slots, slot_team, ball, fs, md.PITCH, md.FPS)
            P, pnames, pkinds = pc.compute(slots, slot_team, ball, fs, md.PITCH, md.FPS)
            graded = np.zeros(len(codes), bool)
            for _, s, e in blocks:
                graded[s:e + 1] = True
            graded &= codes != 3
            idx = np.where(graded)[0]
            idx = idx[(idx % md.STRIDE) == (seed % md.STRIDE)]
            yc = md.change_target(codes)
            same = None
            if a.lde_ds:
                z = np.load(os.path.join(a.lde_ds, "{}_{}.npz".format(match, seed)))
                same = bool(np.array_equal(z["idx"], idx) and
                            np.array_equal(z["X"], F[idx], equal_nan=True))
                assert same, "regenerated degradation differs from LDE's dataset: " + out
            np.savez_compressed(out, X=F[idx], P=P[idx], y=codes[idx], yc=yc[idx], idx=idx,
                                names=np.array(names), kinds=np.array(kinds),
                                pnames=np.array(pnames), pkinds=np.array(pkinds))
            print("{} {} seed {}: {} rows, lde-identical {}, {:.0f}s".format(
                a.variant, match, seed, len(idx), same, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
