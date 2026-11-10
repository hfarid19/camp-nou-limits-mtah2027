#!/usr/bin/env python3
"""Member-alone DFL fine4 (raw codes, no vote rule) on the game-seeds available for every listed member, paired vs
the first: python -I candidates/nn_pointer/dfl_alone.py REF M1 [M2 ...]   (folders under probs/nn_pff_dfl)"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nn_pff"))
import vote_offline as vo  # noqa: E402

P = os.path.join(os.environ["MTAH_WORK"], "probs", "nn_pff_dfl")
names = sys.argv[1:]
info = vo.dfl_info()
units = [u for u in info if all(os.path.isfile(os.path.join(P, m, "dfl2_{}_{}_codes.npy".format(*u))) for m in names)]
S = {m: {u: vo.score_dfl(np.load(os.path.join(P, m, "dfl2_{}_{}_codes.npy".format(*u))).astype(np.int8), info[u][1])
         for u in units} for m in names}
print("units", len(units), "games", len({u[0] for u in units}))
for m in names:
    f = np.mean([S[m][u]["fine4"] for u in units])
    if m == names[0]:
        print("{:14s} fine4 {:.4f}".format(m, f))
        continue
    d = np.array([S[m][u]["fine4"] - S[names[0]][u]["fine4"] for u in units])
    g = {}
    for u, v in zip(units, d):
        g.setdefault(u[0], []).append(v)
    gm = np.array([np.mean(v) for v in g.values()])
    se = gm.std(ddof=1) / np.sqrt(len(gm)) if len(gm) > 1 else float("nan")
    print("{:14s} fine4 {:.4f}  vs {} {:+.4f} ± {:.4f} (wins {}/{})".format(m, f, names[0], d.mean(), se, int((d > 0).sum()), len(d)))
