#!/usr/bin/env python3
"""Length fingerprint, possessions per 90 and dead share of the SoccerTrack v2 label variants.

    python -I candidates/soccertrack/label_stats.py [--dfl]

Fingerprint: spellF1 (pm.evaluate) of a perfect submission shifted k = 3 / 8 / 25 frames late inside
each graded half (candidates/pff/label_stats.py, same code). It depends only on the truth's spell
lengths; EVALUATION.md publishes 0.982 / 0.893 / 0.654 for the full-match truth (~306 / 90),
PFF touch 0.977 / 0.902 / 0.632 (314 / 90), DFL raw flag 0.960 / 0.862 / 0.557 (412 / 90).
Also: share of possession spells shorter than 1 s / 2 s, and agreement of each team reading with
the chosen one on alive frames.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND, os.path.join(os.path.dirname(CAND), "evaluation")]
import label_stats as base         # noqa: E402  (candidates/pff/label_stats.py)
import possession_metrics as pm    # noqa: E402

WORK = base.WORK
sys.path.insert(0, HERE)
import labels as L                 # noqa: E402


def short_share(gt, runs):
    sp = pm.possession_spells(gt, runs)
    d = np.array([s[2] - s[1] if len(s) >= 3 else 0 for s in sp], float) if sp else np.zeros(0)
    return d


def summarise(name, files):
    rows = []
    for f in files:
        z = np.load(f)
        gt, runs = base.gt_blocks(z)
        r = base.stats(gt, runs)
        rows.append(r)
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("{:12s} n={:2d}  shift 3/8/25: {:.3f} / {:.3f} / {:.3f}   poss/90 {:4.0f} (per 90 live {:4.0f})"
          "  dead {:.3f}  mean spell {:4.1f} s  minutes {:5.1f}".format(
              name, len(rows), m["s3"], m["s8"], m["s25"], m["poss90"], m["poss90_alive"], m["dead"],
              m["mean_s"], m["min"]), flush=True)
    return rows


def spell_lengths(files):
    out = []
    for f in files:
        z = np.load(f)
        gt, runs = base.gt_blocks(z)
        for a, b in runs:
            g = gt[a:b]
            ch = np.r_[0, np.where(g[1:] != g[:-1])[0] + 1, len(g)]
            for s, e in zip(ch[:-1], ch[1:]):
                if g[s] in (pm.H, pm.A):
                    out.append((e - s) / 25.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", action="store_true")
    a = ap.parse_args()
    print("{:12s}       shift 3/8/25: {:.3f} / {:.3f} / {:.3f}   (EVALUATION.md full-match truth, ~306/90)".format(
        "truth", *base.TRUTH))
    if a.dfl:
        summarise("DFL raw", sorted(glob.glob(os.path.join(WORK, "idsse", "npz", "*.npz"))))
        summarise("PFF touch", sorted(glob.glob(os.path.join(WORK, "pff_wc2022", "variants", "touch", "PFF-*.npz"))))
    for v in L.VARIANTS:
        files = sorted(glob.glob(os.path.join(WORK, "soccertrack", "variants", v, "ST-*.npz")))
        if files:
            summarise("ST " + v, files)
    print("\nalive spell lengths (s): share < 0.5 / 1 / 2 s, median")
    sets = [("ST " + v, sorted(glob.glob(os.path.join(WORK, "soccertrack", "variants", v, "ST-*.npz"))))
            for v in L.VARIANTS]
    if a.dfl:
        sets = [("DFL raw", sorted(glob.glob(os.path.join(WORK, "idsse", "npz", "*.npz")))),
                ("PFF touch", sorted(glob.glob(os.path.join(WORK, "pff_wc2022", "variants", "touch", "PFF-*.npz"))))] + sets
    for name, files in sets:
        if files:
            d = spell_lengths(files)
            print("{:12s} {:.3f} / {:.3f} / {:.3f}   median {:.1f} s".format(name, np.mean(d < 0.5), np.mean(d < 1),
                                                                          np.mean(d < 2), np.median(d)))
    print("\nteam agreement with {} on frames alive in both:".format(L.CHOSEN))
    ref = sorted(glob.glob(os.path.join(WORK, "soccertrack", "variants", L.CHOSEN, "ST-*.npz")))
    for v in L.VARIANTS:
        acc = []
        for f in ref:
            g = os.path.join(WORK, "soccertrack", "variants", v, os.path.basename(f))
            if not os.path.isfile(g):
                continue
            x, y = np.load(f), np.load(g)
            al = (x["status"] == 1) & (y["status"] == 1)
            acc.append(np.mean(x["poss"][al] == y["poss"][al]))
        if acc:
            print("  {:12s} {:.3f}".format(v, np.mean(acc)))


if __name__ == "__main__":
    main()
