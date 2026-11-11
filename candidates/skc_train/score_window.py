#!/usr/bin/env python3
"""Round-5 "+SKC" secondary reading "10-minutes-out" (prereg addendum 2026-11-10 11:30): score whole-game codes on the
SkillCorner 10-minute windows only, paired vs a reference on the same rows.

    python -I candidates/skc_train/score_window.py NAME_nowin REF [NAME2_nowin ...]

Codes: probs/skc/<name>/skc2_<game>_2027_codes.npy (whole game; bench_skc degraded inputs). Window rows:
$MTAH_WORK/skc_aleague/window10.json ("win", candidates/skc/window10.py). Labels: bench_skc.targets (chosen labels raw
and control.fold T = 0.5 / 1 s, computed on the WHOLE game, then cut to the window); fine3 = mean spellF1 of the three
on the cut. Also: predicted / true possession spells inside the window. Paired: mean difference, SE over the 20 games,
wins. This is an in-game holdout (same teams, same tracking session in training): optimistic for a new match.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "skc"))
import bench_skc as bs  # noqa: E402

pm = bs.pm
WIN = json.load(open(os.path.join(bs.SKC, "window10.json")))["games"]


def score(name, g):
    a, b = WIN["SKC-" + g]["win"]
    pred = np.load(os.path.join(bs.PROBS, name, "skc2_{}_2027_codes.npy".format(g))).astype(np.int8)
    T = bs.targets(g)
    r = {k: pm.evaluate(T[k][a:b], pred[a:b], bootstrap_n=0)["spell_f1_mean"] for k in bs.FINE}
    r["fine3"] = float(np.mean([r[k] for k in bs.FINE]))
    ev = pm.evaluate(T["raw"][a:b], pred[a:b], bootstrap_n=0)
    r["n_pred"], r["n_gt"] = ev["n_pred_spells"], ev["n_gt_spells"]
    return r


def main():
    names, ref = [sys.argv[1]] + sys.argv[3:], sys.argv[2]
    games = bs.all_games()
    R = {n: {g: score(n, g) for g in games} for n in names + [ref]}
    for n in [ref] + names:
        rows = R[n]
        print("{:28s} window fine3 {:.4f}  poss in windows pred {} / true {}".format(
            n, np.mean([r["fine3"] for r in rows.values()]), sum(r["n_pred"] for r in rows.values()),
            sum(r["n_gt"] for r in rows.values())))
    for n in names:
        for k in ("fine3", "raw", "T0.5", "T1"):
            d = np.array([R[n][g][k] - R[ref][g][k] for g in games])
            print("  {} - {} {:6s} {:+.4f} ± {:.4f} (wins {}/{}, losses {})".format(
                n, ref, k, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d), int((d < 0).sum())))


if __name__ == "__main__":
    main()
