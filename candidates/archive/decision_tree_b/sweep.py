#!/usr/bin/env python3
"""Depth selection for the two-stage rule model on the DFL bench, leave-one-game-out, nested.

    OMP_NUM_THREADS=1 python candidates/decision_tree_b/sweep.py [--jobs 2]

Scores what bench.py --dfl2 scores (bench.dfl2_prepared games, seeds 2027-2029,
possession_metrics.evaluate, controlled_labels T = 1, 2, 3) for every D<d>_T<t> setting in
$MTAH_WORK/arch/decision_tree_b/sweep, always with fold_rules_<gi>.npz on game gi. Also scores a
no-decoder baseline (per-frame argmax, then the 2 s control fold). Writes sweep_results.json and
prints the per-setting means plus the nested estimate: for each game, the setting that is best on
the other six games, scored on that game.
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
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
ARCH = os.path.join(WORK, "arch", "decision_tree_b")


def setup():
    for p in (CAND, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)


def score_game(args):
    gi, match, seed, settings, sweep_dir = args
    setup()
    import bench
    import control
    import pipeline as dtb
    pm = bench.pm
    slots, slot_team, ball, gt, blocks, sign = bench.dfl2_prepared(match, seed)
    F, names, kinds = dtb.fe.compute(slots, slot_team, ball, dtb.fe.frame_sign(len(ball), sign, blocks),
                                     bench.tsd.PITCH, 25)          # no memo: fresh per game-seed
    lg = dtb.dec.long_gap_gated(slots, ball, 25, dtb.dec.CFG["gap_s"], dtb.dec.CFG["gap_speed_ms"])
    cg = [bench.controlled_labels(gt, T) for T in (1, 2, 3)]
    runs = [(int(a), int(b) + 1) for _, a, b in blocks]
    graded = np.zeros(len(gt), bool)
    for a, b in runs:
        graded[a:b] = True
    out = []
    for s in settings:
        rules = dtb.load_rules(os.path.join(ARCH, sweep_dir, s, "fold_rules_{}.npz".format(gi)))
        p = dtb.proba_from(rules, F, names, kinds)
        pred = dtb.decode(p, F, names, lg, {}, blocks, len(gt), 25, control_s=2.0)
        am = np.where(graded, np.argmax(p, 1), pm.X).astype(np.int8)
        am = control.fold(am, 2.0, runs, 25)
        row = dict(setting=s, game=match, seed=seed)
        for tag, codes in (("dec", pred), ("argmax", am)):
            ctrls = [pm.evaluate(g, codes, bootstrap_n=0)["spell_f1_mean"] for g in cg]
            r = pm.evaluate(gt, codes, bootstrap_n=0)
            row[tag] = dict(ctrl_mean=float(np.mean(ctrls)), ctrl=ctrls, raw=r["spell_f1_mean"],
                            macro=r["macro_f1"], poss=int(r["n_pred_spells"]))
        out.append(row)
    print("  {} {} done".format(match, seed), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--settings", nargs="*", default=None)
    ap.add_argument("--dir", default="sweep", help="sub-folder of $MTAH_WORK/arch/decision_tree_b")
    a = ap.parse_args()
    setup()
    import bench
    matches = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    settings = a.settings or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ARCH, a.dir, "D*_T*")))
    jobs = [(gi, m, seed, settings, a.dir) for gi, m in enumerate(matches) for seed in bench.DFL2_SEEDS]
    rows = []
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for r in pool.imap_unordered(score_game, jobs):
            rows.extend(r)
    with open(os.path.join(ARCH, a.dir + "_results.json"), "w") as handle:
        json.dump(rows, handle)
    games = sorted({r["game"] for r in rows})

    def per_game(s, tag="dec", key="ctrl_mean"):
        return np.array([np.mean([r[tag][key] for r in rows if r["setting"] == s and r["game"] == g])
                         for g in games])
    print("{:8s} {:>9s} {:>8s} {:>8s} {:>6s} {:>14s}".format("setting", "ctrl_mean", "raw", "macroF1",
                                                             "poss", "argmax+fold"))
    for s in settings:
        rr = [r for r in rows if r["setting"] == s]
        print("{:8s} {:9.4f} {:8.4f} {:8.4f} {:6d} {:14.4f}".format(
            s, per_game(s).mean(), per_game(s, key="raw").mean(), per_game(s, key="macro").mean(),
            int(sum(r["dec"]["poss"] for r in rr) / 3), per_game(s, "argmax").mean()))
    T = {s: per_game(s) for s in settings}
    nested = []
    for i, g in enumerate(games):
        others = [j for j in range(len(games)) if j != i]
        pick = max(settings, key=lambda s: T[s][others].mean())
        nested.append(T[pick][i])
        print("  nested: {} -> {} {:.4f}".format(g, pick, T[pick][i]))
    best = max(settings, key=lambda s: T[s].mean())
    print("in-sample best {} {:.4f}; nested {:.4f}".format(best, T[best].mean(), np.mean(nested)))


if __name__ == "__main__":
    main()
