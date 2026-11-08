#!/usr/bin/env python3
"""Depth / min_samples_leaf selection on the DFL bench, leave-one-game-out, without rerunning
bench.py once per setting.

    OMP_NUM_THREADS=2 python candidates/decision_tree/sweep.py [--jobs 2]

Scores exactly what bench.py --dfl2 scores (the same degrade_v2 games and seeds 2027-2029 via
bench.dfl2_prepared, the same possession_metrics.evaluate and controlled_labels), but computes
each game's features once (cached in $MTAH_WORK/arch/decision_tree/cache) and then decodes every
tree setting from $MTAH_WORK/arch/decision_tree/sweep/L<leaf>_d<depth>/fold_tree_<gi>.npz, the fold
that never saw game gi, with pipeline.decode (control_s 0 = raw and CONTROL_S). Writes
sweep_results.json next to the trees. The chosen setting is then confirmed with bench.py itself.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path.insert(0, CAND)
sys.path.insert(0, HERE)
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
ARCH = os.path.join(WORK, "arch", "decision_tree")


def cached(match, seed):
    import bench
    import pipeline as dtp
    path = os.path.join(ARCH, "cache", "{}_{}.npz".format(match, seed))
    slots, slot_team, ball, gt, blocks, sign = bench.dfl2_prepared(match, seed)
    if not os.path.isfile(path):
        F, names, kinds = dtp.features(slots, slot_team, ball, sign, blocks, bench.tsd.PITCH, bench.tsd.FPS)
        lg = dtp.dec.long_gap_gated(slots, ball, bench.tsd.FPS, dtp.dec.CFG["gap_s"], dtp.dec.CFG["gap_speed_ms"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, F=F, names=np.array(names), kinds=np.array(kinds), long_gap=lg)
    z = np.load(path)
    return (z["F"], [str(s) for s in z["names"]], [str(s) for s in z["kinds"]], z["long_gap"],
            gt, blocks)


def score_game(args):
    gi, match, seed, settings = args
    import bench
    import pipeline as dtp
    pm = bench.pm
    F, names, kinds, lg, gt, blocks = cached(match, seed)
    ctrl_gt = [bench.controlled_labels(gt, T) for T in (1, 2, 3)]
    out = []
    for s in settings:
        z = np.load(os.path.join(ARCH, "sweep", s, "fold_tree_{}.npz".format(gi)))
        tree = {k: z[k] for k in z.files}
        tree["names"] = [str(v) for v in tree["names"]]
        p = dtp.sym_proba(tree, F, names, kinds)
        for cs in (0.0, dtp.CONTROL_S):
            pred = dtp.decode(p, F, names, lg, {}, blocks, len(gt), 25, control_s=cs)
            r = pm.evaluate(gt, pred, bootstrap_n=0)
            ctrls = [pm.evaluate(g, pred, bootstrap_n=0)["spell_f1_mean"] for g in ctrl_gt]
            out.append(dict(setting=s, control_s=cs, game=match, seed=seed,
                            spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                            ctrl_mean=float(np.mean(ctrls)), ctrl=ctrls, poss=int(r["n_pred_spells"]),
                            gt_poss=int(r["n_gt_spells"])))
    print("  {} {} done".format(match, seed), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--settings", nargs="*", default=None)
    a = ap.parse_args()
    import bench
    matches = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    settings = a.settings or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ARCH, "sweep", "L*_d*")))
    jobs = [(gi, m, seed, settings) for gi, m in enumerate(matches) for seed in bench.DFL2_SEEDS]
    rows = []
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for r in pool.imap_unordered(score_game, jobs):
            rows.extend(r)
    with open(os.path.join(ARCH, "sweep_results.json"), "w") as handle:
        json.dump(rows, handle)
    print("{:12s} {:>4s} {:>8s} {:>8s} {:>8s} {:>6s}".format("setting", "ctrl", "ctrl_mean", "spellF1", "macroF1", "poss"))
    for s in settings:
        for cs in (0.0, 2.0):
            rr = [r for r in rows if r["setting"] == s and r["control_s"] == cs]
            print("{:12s} {:>4g} {:8.4f} {:8.4f} {:8.4f} {:6d}".format(
                s, cs, np.mean([r["ctrl_mean"] for r in rr]), np.mean([r["spellF1"] for r in rr]),
                np.mean([r["macroF1"] for r in rr]), int(sum(r["poss"] for r in rr) / 3)))


if __name__ == "__main__":
    main()
