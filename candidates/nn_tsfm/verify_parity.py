#!/usr/bin/env python3
"""numpy vs torch-CPU parity of the nn_tsfm forward on full PFF bench games.

    python candidates/nn_tsfm/verify_parity.py <weights.npz> GAME [GAME ...]

For each game: features of the degraded bench input (bench_pff.pff2_prepared), both orientations,
all 5 phases, windowed exactly as pipeline.model_probs does; reports max |dP| on the (N, 3) class
probabilities and on the hand-over probability, the max |dlogit| on the raw window logits of the
first phase, the number of frames whose argmax differs, and wall time per backend.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND, ROOT, os.path.join(ROOT, "evaluation"), HERE]
os.environ.setdefault("NN_TSFM_DEVICE", "cpu")
import bench_pff                      # noqa: E402
import tune_on_sample_data as tsd     # noqa: E402
import pipeline as P                  # noqa: E402
import moment_np                      # noqa: E402
import features as fe                 # noqa: E402


def main():
    wpath = os.path.abspath(sys.argv[1])
    games = sys.argv[2:]
    W = moment_np.load(wpath)
    for g in games:
        slots, slot_team, ball, gt, blocks, sign = bench_pff.pff2_prepared(g)
        sg = fe.frame_sign(len(ball), sign, blocks)
        F, names, kinds = fe.compute(slots, slot_team, ball, sg, tsd.PITCH, tsd.FPS)
        res = {}
        for be in ("numpy", "torch"):
            P.BACKEND = be
            t0 = time.time()
            Pc, pc = P.model_probs(F, names, kinds, blocks, W)
            res[be] = (Pc, pc, time.time() - t0)
        # raw logits of one window, first phase, both backends
        first, last = blocks[0][1], blocks[0][2]
        Z = moment_np.prep(F[first:last + 1:P.STRIDE], W["mu"], W["sd"], W["ind"])[:P.WIN]
        v = np.ones(len(Z), bool)
        ln = moment_np.forward(W, Z, v)
        lt = P._torch_forward(W)(Z, v)
        dP = np.abs(res["numpy"][0] - res["torch"][0]).max()
        dc = np.abs(res["numpy"][1] - res["torch"][1]).max()
        dl = np.abs(ln - lt).max()
        na = int((res["numpy"][0].argmax(1) != res["torch"][0].argmax(1)).sum())
        print("{}: frames {} max|dP| {:.2e} max|dpc| {:.2e} max|dlogit| {:.2e} argmax diffs {} | numpy {:.0f}s torch-cpu {:.0f}s".format(
            g, len(ball), dP, dc, dl, na, res["numpy"][2], res["torch"][2]), flush=True)


if __name__ == "__main__":
    main()
