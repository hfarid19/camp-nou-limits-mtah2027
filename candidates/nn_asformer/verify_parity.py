#!/usr/bin/env python3
"""Numpy (asf_np.py) vs torch-CPU (model.py) parity of an exported model on full games.

    PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 python -I candidates/nn_asformer/verify_parity.py MODEL.npz [GAME ...]

Inputs = the bench's degraded PFF games (candidates/pff/bench_pff.pff2_prepared, seed 2027) run
through the pipeline's own feature code; every 5 fps phase of every half is a sequence (the exact
inference path of pipeline.py). Reports max |P_numpy - P_torch| over class and boundary
probabilities per game, plus torch MPS if available. Default games: 10504, 3816 (selection games;
no labels are read, nothing is scored).
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [HERE, os.path.join(CAND, "pff"), CAND, ROOT, os.path.join(ROOT, "evaluation")]
import asf_np     # noqa: E402
import model as M  # noqa: E402
import features as fe  # noqa: E402
import bench_pff  # noqa: E402
import tune_on_sample_data as tsd  # noqa: E402


def main():
    path = sys.argv[1]
    games = sys.argv[2:] or ["10504", "3816"]
    z = np.load(path)
    net = asf_np.Net(z)
    cfg = dict(M.CFG, **json.loads(str(z["cfg"])))
    tnet = M.Net(len(net.names), cfg)
    tnet.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("w/")})
    tnet.eval()
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    worst = {d: 0.0 for d in devs}
    for g in games:
        slots, slot_team, ball, _, blocks, sign = bench_pff.pff2_prepared(g)
        sgn = fe.frame_sign(len(ball), sign, blocks)
        F, names, kinds = fe.compute(slots, slot_team, ball, sgn, tsd.PITCH, tsd.FPS)
        assert names == net.names
        n_steps, t_np, t_t = 0, 0.0, 0.0
        for _, first, last in blocks:
            for r in range(5):
                fr = np.arange(first + r, last + 1, 5)
                Xn = net.normalise(F[fr].astype(np.float64))
                t0 = time.time()
                pc, pb = net.forward(Xn)
                t_np += time.time() - t0
                n_steps += len(fr)
                xt = torch.from_numpy(Xn.astype(np.float32))[None]
                for d in devs:
                    t0 = time.time()
                    with torch.no_grad():
                        cs, bs = tnet.to(d)(xt.to(d))
                    t_t += time.time() - t0 if d == "cpu" else 0
                    pct = torch.softmax(cs[-1], -1)[0].cpu().numpy()
                    pbt = torch.sigmoid(bs[-1])[0].cpu().numpy()
                    worst[d] = max(worst[d], float(np.abs(pct - pc).max()), float(np.abs(pbt - pb).max()))
        print("game {}: {} steps, numpy {:.1f}s, torch cpu {:.1f}s; running max |diff| {}".format(
            g, n_steps, t_np, t_t, {d: "{:.2e}".format(v) for d, v in worst.items()}), flush=True)
    print("MAX |numpy - torch| over {} games: {}".format(len(games), {d: "{:.2e}".format(v) for d, v in worst.items()}))


if __name__ == "__main__":
    main()
