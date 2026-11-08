"""Regret table across label readings, possessions per 90 and share of short spells, for saved codes dirs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, fastscore, control
import possession_metrics as pm

W = data.W
DIRS = {"shipped fold2": data.SHIPPED, "v3b joint (nested)": W + "/ensemble_ctrl_v3b_joint",
        "v3b delta iter (nested)": W + "/ensemble_ctrl_v3b_delta_iter",
        "v3b evtavg iter (nested, primary)": W + "/ensemble_ctrl_v3b_evtavg_iter",
        "durfold (nested T)": W + "/ensemble_ctrl_v3b_durfold"}
READ = ["raw", "T1", "T1.5", "T2", "T2.5", "T3", "evt3", "evt2"]
FOLDS = [0, 1, 1.5, 2, 2.5, 3]
S = {k: {r: [] for r in READ} for k in list(DIRS) + ["fold{:g}".format(t) for t in FOLDS]}
short, nsp, mins, true_short, true_n = {k: 0 for k in S}, {k: 0 for k in S}, 0.0, {r: 0 for r in READ}, {r: 0 for r in READ}
for match in data.MATCHES:
    for seed in data.SEEDS:
        nm = data.name(match, seed)
        _, _, _, gt, blocks, _ = data.bench.dfl2_prepared(match, seed)
        tg = data.targets(match, gt)
        T = {r: fastscore.Target(tg[r]) for r in READ}
        runs = [(f, l + 1) for _, f, l in blocks]
        mins += sum(l + 1 - f for f, l in runs) / 25 / 60
        base = np.load(data.FOLD0 + "/" + nm + "_codes.npy")
        codes = {k: np.load(d + "/" + nm + "_codes.npy") for k, d in DIRS.items()}
        for t in FOLDS:
            codes["fold{:g}".format(t)] = control.fold(base, t, runs, 25)
        for k, c in codes.items():
            for r in READ:
                S[k][r].append(T[r].f1(c)[0])
            sp = pm.possession_spells(c, runs)
            nsp[k] += len(sp); short[k] += sum(b - a < 50 for a, b, _ in sp)
        for r in READ:
            sp = pm.possession_spells(tg[r], pm.valid_runs(tg[r]))
            true_n[r] += len(sp); true_short[r] += sum(b - a < 50 for a, b, _ in sp)
mean = {k: {r: np.mean(v) for r, v in S[k].items()} for k in S}
bestf = {r: max(mean["fold{:g}".format(t)][r] for t in FOLDS) for r in READ}
print("spellF1 by label reading; regret = best vote fold (0-3 s) at that reading minus score")
print("{:36s}".format("") + "".join("{:>14s}".format(r) for r in READ) + "   max regret  mean")
for k in S:
    reg = [bestf[r] - mean[k][r] for r in READ]
    print("{:36s}".format(k) + "".join("{:7.4f}/{:6.4f}".format(mean[k][r], reg[i]) for i, r in enumerate(READ))
          + "   {:.4f}  {:.4f}".format(max(reg), np.mean(reg)))
print("\npaired diff vs shipped by reading (SE over 7 game means):")
for k in DIRS:
    if k == "shipped fold2":
        continue
    line = "{:36s}".format(k)
    for r in READ:
        d = np.array(S[k][r]) - np.array(S["shipped fold2"][r])
        gm = d.reshape(7, 3).mean(1)
        line += "{:+.4f}±{:.4f} ".format(gm.mean(), gm.std(ddof=1) / np.sqrt(7))
    print(line)
print("\npossessions per 90 and share of spells < 2 s (graded minutes per game-seed {:.1f})".format(mins / 21))
for k in S:
    print("  {:36s} {:6.1f}/90  short {:.3f}".format(k, nsp[k] / mins * 90, short[k] / nsp[k]))
for r in READ:
    print("  truth {:30s} {:6.1f}/90  short {:.3f}".format(r, true_n[r] / mins * 90, true_short[r] / true_n[r]))
