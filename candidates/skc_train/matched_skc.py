#!/usr/bin/env python3
"""Round-5 "+SKC" supplementary check (not part of the gate): SkillCorner fine3 of a candidate vs its parent re-folded
to the candidate's possession count per game (mbr_decode/analysis.matched: merge_short 1 s + control.fold T, T chosen
on the grid to match the count), so a gain from emitting fewer spells is not counted.

    python -I candidates/skc_train/matched_skc.py CAND PARENT
Codes: probs/skc/<name>/skc2_<game>_2027_codes.npy.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mbr_decode"))
import analysis  # noqa: E402
import data      # noqa: E402
import mbr       # noqa: E402


def codes(name, g):
    return np.load(os.path.join(data.PROBS, "skc", name, "skc2_{}_2027_codes.npy".format(g))).astype(np.int8)


def main():
    cand, par = sys.argv[1], sys.argv[2]
    a, b, m, nc, npar = {}, {}, {}, 0, 0
    for g in data.units("skc"):
        blocks, tg = analysis.targets("skc", g)
        runs = mbr.runs_of(blocks)
        c, p = codes(cand, g), codes(par, g)
        n = analysis.nsp(c, runs)
        nc, npar = nc + n, npar + analysis.nsp(p, runs)
        a[g] = analysis.scores("skc", c, tg)["main"]
        b[g] = analysis.scores("skc", p, tg)["main"]
        m[g] = analysis.matched("skc", [p], blocks, n, tg)["main"]
    for lab, ref in (("parent as is", b), ("parent at matched count", m)):
        d = np.array([a[g] - ref[g] for g in a])
        print("{} vs {} ({}): {:+.4f} ± {:.4f} (wins {}/{})".format(cand, par, lab, d.mean(), d.std(ddof=1) / np.sqrt(len(d)),
                                                                 int((d > 0).sum()), len(d)))
    print("spells: candidate {} parent {}".format(nc, npar))


if __name__ == "__main__":
    main()
