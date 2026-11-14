"""Metrica-trained stacker S2 on v8's four members (round-5 prereg addendum of 13 Nov 2026 09:02 CST, "S2a").

Inputs per frame: the (H, A, D) probabilities of the four F4sP members (slot order of candidates/final_f4sp/pipeline.py:
TCN, MS-TCN, radar, pointer), rounded to float32, at offsets -2, -1, -0.4, 0, +0.4, +1, +2 s (frames -50, -25, -10, 0, +10,
+25, +50 at 25 fps), each source index clamped inside the frame's graded block -> 4 x 7 x 3 = 84 predictors (member-major,
then offset, then class). Model: LightGBM multiclass, 200 trees per class, max depth 4, learning rate 0.05
(stacker_s2.txt; trained by train_stacker.py on strict Metrica labels).

Applied exactly as the experiment's scoring did (arch/r5_arch/stack_aug/code/stack.py, predict): at EVERY frame inside
the graded blocks (25 per second; training used every 5th frame only), probabilities = the booster's softmax output
(float64), codes = their argmax; outside the blocks probabilities 1/3 each and code 3 (X). The probabilities are returned
rounded to float32, as the experiment saved them before decoding. Prediction is single-threaded (num_threads=1).
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "stacker_s2.txt")
META = os.path.join(HERE, "stacker_s2_meta.json")
OFFS = np.array([-50, -25, -10, 0, 10, 25, 50])


def runs_of(blocks):
    return [(int(f), int(l) + 1) for _, f, l in blocks]


def frame_index(n, runs):
    """(n, 7) source frame per frame and offset, clamped inside the frame's block (whole array outside blocks)."""
    t = np.arange(n)
    lo, hi = np.zeros(n, np.int64), np.full(n, n - 1, np.int64)
    for a, b in runs:
        lo[a:b], hi[a:b] = a, b - 1
    return np.clip(t[:, None] + OFFS[None, :], lo[:, None], hi[:, None])


def inside(n, runs):
    m = np.zeros(n, bool)
    for a, b in runs:
        m[a:b] = True
    return m


def feats(P, runs, rows=None, swap=False):
    """84 predictors (float32) for the given rows; swap exchanges the H and A columns of every member and offset."""
    n = len(P[0])
    J = frame_index(n, runs)
    if rows is not None:
        J = J[rows]
    cols = []
    for p in P:
        q = p[:, [1, 0, 2]] if swap else p
        for k in range(len(OFFS)):
            cols.append(q[J[:, k]])
    return np.concatenate(cols, 1).astype(np.float32)          # member-major, offset, class


def load():
    """A fresh Booster of the shipped model (LightGBM version and model sha256 checked against stacker_s2_meta.json).
    Not cached here: the caller keeps it (so bench.py --dfl, which swaps lightgbm.Booster per DFL game, sees every load)."""
    import lightgbm as lgb
    with open(META) as handle:
        meta = json.load(handle)
    assert lgb.__version__ == meta["lgb_version"], "lightgbm version differs from training"
    with open(MODEL, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == meta["sha256"], "stacker_s2.txt differs from its meta"
    return lgb.Booster(model_file=MODEL)


def predict(P, blocks, model):
    """P: the four members' (N, 3) probabilities (float32 values); model: load()'s Booster.
    -> (probs (N, 3) float32, codes (N,) int8)."""
    P = [np.asarray(p, np.float32).astype(np.float64) for p in P]
    b = model
    n = len(P[0])
    runs = runs_of(blocks)
    ins = inside(n, runs)
    pr = np.full((n, 3), 1.0 / 3, np.float64)
    if ins.any():
        pr[ins] = b.predict(feats(P, runs, np.where(ins)[0]), num_threads=1)
    codes = np.full(n, 3, np.int8)
    codes[ins] = np.argmax(pr[ins], 1)
    return pr.astype(np.float32), codes
