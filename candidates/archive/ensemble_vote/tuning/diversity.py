"""Members' DFL scores and pairwise frame-error correlation (phi of the per-frame error indicators,
graded frames, pooled over the 7 DFL games), from the collected outputs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitlib as F   # noqa: E402


def main():
    sp = F.Split(sys.argv[1] if len(sys.argv) > 1 else "dfl", F.members.ALL)
    order = sorted(sp.slugs, key=lambda s: -np.mean([v["spell"] for v in sp.member_scores[s].values()]))
    err = {}
    for s in order:
        e = []
        for t in sp.tags:
            g = sp.gt[t]
            m = g != 3
            e.append(sp.codes[(s, t)][m] != g[m])
        err[s] = np.concatenate(e).astype(float)
        sc = sp.member_scores[s]
        print("{:22s} spell {:.4f} macro {:.4f} poss {} frame-err {:.4f}".format(
            s, np.mean([v["spell"] for v in sc.values()]), np.mean([v["macro"] for v in sc.values()]),
            sum(v["n_pred"] for v in sc.values()), err[s].mean()))
    short = [s[:6] for s in order]
    print(" " * 23 + " ".join("{:>6s}".format(x) for x in short))
    for a in order:
        print("{:22s} ".format(a) + " ".join("{:6.2f}".format(np.corrcoef(err[a], err[b])[0, 1])
                                            for b in order))


if __name__ == "__main__":
    main()
