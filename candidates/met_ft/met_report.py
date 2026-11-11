#!/usr/bin/env python3
"""MET-FT Metrica leave-one-game-out report: per left-out game (mean over the 2 bench seeds), the fine-tuned member
(trained on the other Metrica games) vs its parent, on the strict labels (PRIMARY) and the practice labels; gate part 1
of the prereg (better on EVERY left-out game and mean gain >= +0.015, primary labels).

    python -I candidates/met_ft/met_report.py [m5c ptr]
reads $MTAH_WORK/arch/r5_arch/met_ft/scores/parent_<m>.jsonl and <m>_logo_G<k>.jsonl
"""
import json
import os
import sys

import numpy as np

SD = os.path.join(os.environ["MTAH_WORK"], "arch", "r5_arch", "met_ft", "scores")
GAMES = ["G1", "G2", "G3"]


def rows(name):
    p = os.path.join(SD, name + ".jsonl")
    out = {}
    if os.path.isfile(p):
        for line in open(p):
            r = json.loads(line)
            out[(r["game"], r["seed"])] = r       # last row wins
    return out


def report(m):
    par = rows("parent_" + m)
    print("== {} (FT on the other Metrica games, scored on the left-out one; mean of seeds 2027/2028)".format(m))
    print("{:4s} {:>9s} {:>9s} {:>8s} | {:>9s} {:>9s} {:>8s} | {:>6s} {:>6s} {:>6s}".format(
        "game", "par strict", "FT strict", "gain", "par prac", "FT prac", "gain", "p90par", "p90FT", "p90gt"))
    gains, gp, done = [], [], True
    for g in GAMES:
        ft = rows("{}_logo_{}".format(m, g))
        ks = [(g, s) for s in (2027, 2028)]
        if not all(k in par and k in ft for k in ks):
            print("{:4s} pending".format(g))
            done = False
            continue
        ps, fs = np.mean([par[k]["strict"] for k in ks]), np.mean([ft[k]["strict"] for k in ks])
        pp, fp = np.mean([par[k]["practice"] for k in ks]), np.mean([ft[k]["practice"] for k in ks])
        gains.append(fs - ps)
        gp.append(fp - pp)
        print("{:4s} {:9.4f} {:9.4f} {:+8.4f} | {:9.4f} {:9.4f} {:+8.4f} | {:6.0f} {:6.0f} {:6.0f}".format(
            g, ps, fs, fs - ps, pp, fp, fp - pp, np.mean([par[k]["p90"] for k in ks]),
            np.mean([ft[k]["p90"] for k in ks]), np.mean([par[k]["p90_strict"] for k in ks])))
    if gains:
        print("mean gain strict {:+.4f} (games better {}/{}), practice {:+.4f} (better {}/{})".format(
            np.mean(gains), sum(x > 0 for x in gains), len(gains), np.mean(gp), sum(x > 0 for x in gp), len(gp)))
    if done:
        ok = all(x > 0 for x in gains) and np.mean(gains) >= 0.015
        print("GATE part 1 (held-out Metrica, strict labels): {}".format("PASS" if ok else "FAIL"))
        return ok
    return None


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["m5c", "ptr"]):
        report(m)
        print()
