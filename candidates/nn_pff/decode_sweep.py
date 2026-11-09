#!/usr/bin/env python3
"""Re-tune the MS-TCN member's ASRF decoder on saved PFF out-of-fold probabilities (nested by fold).

    python candidates/nn_pff/decode_sweep.py NAME [--crit fine3|raw]

NAME = a run_pff.py output folder (probs/nn_pff/NAME with pff2_<g>_2027.npy = class probs and
_pb.npy = boundary probs, both from out-of-fold models). Every grid point decodes all 48 selection
games with pipeline.decode(pc, pb, blocks, n, fps, cfg); the nested score picks the grid point on
the other 3 folds for each fold. Prints the default decoder, the top grid points and the nested result.
"""

import argparse
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vote_offline as vo  # noqa: E402

GRID = dict(b_thr=[0.1, 0.15, 0.2, 0.3, 0.4], peak_win_s=[0.2, 0.4, 0.8], contrast_beta=[0.0, 1.25],
            min_poss_s=[0.0, 0.5, 1.0])


def load_pipeline():
    spec = importlib.util.spec_from_file_location("nnpff_dec", os.path.join(HERE, "pipeline.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--crit", default="fine3")
    a = ap.parse_args()
    pl = load_pipeline()
    d = os.path.join(vo.WORK, "probs", "nn_pff", a.name)
    info = vo.pff_info()
    fold = vo.pff_games()
    keys = list(GRID)
    grid = [dict(zip(keys, v)) for v in itertools.product(*[GRID[k] for k in keys])]
    S = {}
    for g, (blocks, tg) in sorted(info.items()):
        pc = np.load(os.path.join(d, "pff2_{}_2027.npy".format(g))).astype(np.float64)
        pb = np.load(os.path.join(d, "pff2_{}_2027_pb.npy".format(g))).astype(np.float64)
        for i, cfg in enumerate(grid):
            pred = pl.decode(pc, pb, blocks, len(pc), 25, cfg=dict(cfg, min_dead_s=cfg["min_poss_s"]))
            S[(i, g)] = vo.score_pff(pred, tg)
    gs = sorted(info)
    mean = {i: np.mean([S[(i, g)][a.crit] for g in gs]) for i in range(len(grid))}
    default = next(i for i, c in enumerate(grid) if c == dict(b_thr=0.2, peak_win_s=0.4, contrast_beta=1.25, min_poss_s=0.0))
    print("default decoder: {} {:.4f} raw {:.4f}".format(grid[default], mean[default], np.mean([S[(default, g)]["raw"] for g in gs])))
    for i in sorted(mean, key=lambda i: -mean[i])[:10]:
        print("  {} {} {:.4f} raw {:.4f} poss {}".format(i, grid[i], mean[i], np.mean([S[(i, g)]["raw"] for g in gs]),
                                                        sum(S[(i, g)]["n_pred"] for g in gs)))
    res, picks = {}, {}
    for k in range(4):
        tr = [g for g in gs if fold[g] != k]
        best = max(range(len(grid)), key=lambda i: np.mean([S[(i, g)][a.crit] for g in tr]))
        picks[k] = grid[best]
        for g in gs:
            if fold[g] == k:
                res[g] = S[(best, g)]
    dd = np.array([res[g][a.crit] - S[(default, g)][a.crit] for g in gs])
    print("nested {}: {:.4f} vs default {:.4f}: {:+.4f} ± {:.4f} ({}/{})".format(
        a.crit, np.mean([res[g][a.crit] for g in gs]), mean[default], dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)),
        int((dd > 0).sum()), len(dd)))
    print("picks", json.dumps(picks))


if __name__ == "__main__":
    main()
