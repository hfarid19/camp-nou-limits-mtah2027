#!/usr/bin/env python3
"""Missed-spell breakdown on PFF out-of-fold codes: for every ground-truth possession spell (H or A, touch
labels) with no predicted spell of the same label at IoU >= 0.5, the majority predicted label over its frames:
same team (boundary / fragmentation), other team ("wrong team"), dead.

    python -I candidates/nn_pointer/wrong_team.py NAME [NAME ...]     (folders under probs/nn_pff)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND, os.path.dirname(CAND), os.path.join(os.path.dirname(CAND), "evaluation")]
import bench_pff  # noqa: E402

WORK = os.environ["MTAH_WORK"]


def runs(c):
    ch = np.concatenate(([0], np.where(c[1:] != c[:-1])[0] + 1, [len(c)]))
    return [(a, b, int(c[a])) for a, b in zip(ch[:-1], ch[1:])]


def breakdown(gt, pred):
    pr = {}
    for a, b, l in runs(pred):
        pr.setdefault(l, []).append((a, b))
    out = dict(n=0, missed=0, same=0, wrong=0, dead=0)
    for a, b, l in runs(gt):
        if l not in (0, 1):
            continue
        out["n"] += 1
        hit = False
        for c, d in pr.get(l, []):
            if d <= a or c >= b:
                continue
            if (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) >= 0.5:
                hit = True
                break
        if hit:
            continue
        out["missed"] += 1
        cnt = np.bincount(pred[a:b].astype(int), minlength=4)[:3]
        k = int(np.argmax(cnt))
        out["same" if k == l else "dead" if k == 2 else "wrong"] += 1
    return out


def main():
    games = sorted(f.split("_")[1] for f in os.listdir(os.path.join(WORK, "probs", "nn_pff", sys.argv[1])) if f.endswith("_codes.npy"))
    gts = {g: bench_pff.pff2_prepared(g)[3] for g in games}
    for name in sys.argv[1:]:
        tot = dict(n=0, missed=0, same=0, wrong=0, dead=0)
        for g in games:
            f = os.path.join(WORK, "probs", "nn_pff", name, "pff2_{}_2027_codes.npy".format(g))
            if not os.path.isfile(f):
                continue
            r = breakdown(np.asarray(gts[g]), np.load(f))
            for k in tot:
                tot[k] += r[k]
        m = max(tot["missed"], 1)
        print("{:16s} spells {} missed {} ({:.3f}) | of missed: wrong team {:.3f}, same team {:.3f}, dead {:.3f}".format(
            name, tot["n"], tot["missed"], tot["missed"] / max(tot["n"], 1), tot["wrong"] / m, tot["same"] / m, tot["dead"] / m))


if __name__ == "__main__":
    main()
