"""Offline data access for train.py: degraded inputs, the unfolded vote codes, members' saved LOGO
codes and P(H, A, D), and the round-2 targets. Labels are used only as training / scoring targets."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
for _p in (ROOT, os.path.join(ROOT, "evaluation"), CAND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bench                      # noqa: E402
import possession_metrics as pm   # noqa: E402
import spellctl                   # noqa: E402

W = bench.WORK + "/probs"
MEMBER_DIRS = [W + "/temporal_tcn_v2_ctrl", W + "/handover_mstcn_v2_ctrl",
               W + "/ensemble_ctrl_v2/members/ball_radar"]      # vote order (fold_sweep.py)
FOLD0 = W + "/_sweep/fold0"
SHIPPED = W + "/ensemble_ctrl_v2"
MATCHES = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
SEEDS = bench.DFL2_SEEDS
EVT = {"evt3": os.path.join(bench.WORK, "ctrl_events"),
       "evt2": os.path.join(bench.WORK, "arch", "control_events", "variants", "hold2")}


def name(match, seed):
    return "dfl2_{}_{}".format(match, seed)


def game(match, seed):
    slots, slot_team, ball, gt, blocks, sign = bench.dfl2_prepared(match, seed)
    nm = name(match, seed)
    mc = [np.load(os.path.join(d, nm + "_codes.npy")) for d in MEMBER_DIRS]
    mp = [np.load(os.path.join(d, nm + ".npy")) for d in MEMBER_DIRS]
    F = spellctl.Frames(slots, slot_team, ball, sign, blocks, mc, mp)
    base = np.load(os.path.join(FOLD0, nm + "_codes.npy")).astype(np.int8)
    return F, base, blocks, gt


def targets(match, gt):
    out = {"raw": gt}
    for T in (0.5, 1, 1.5, 2, 2.5, 3):
        out["T{:g}".format(T)] = bench.controlled_labels(gt, T)
    for k, d in EVT.items():
        out[k] = np.load(os.path.join(d, match + ".npy"))
    return out
