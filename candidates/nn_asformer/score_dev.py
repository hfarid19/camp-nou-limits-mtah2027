#!/usr/bin/env python3
"""Score saved Metrica dev codes (bench.py default split, --save-probs DIR) paired against a reference.

    python candidates/nn_asformer/score_dev.py DIR [DIR ...] --ref DIR

DIR holds dev_<game>_<seed>_codes.npy (Sample Games 1 and 2, seeds 2027 / 2028: the bench's dev
split; Sample Game 3 is never touched). Prints spellF1 / macroF1 means and the paired difference
(mean, SE over the 4 game-seeds, wins).
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [CAND, os.path.dirname(CAND), os.path.join(os.path.dirname(CAND), "evaluation")]
import bench                     # noqa: E402
import possession_metrics as pm  # noqa: E402


def score_dir(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "dev_*_codes.npy"))):
        name, seed = os.path.basename(f)[4:-len("_codes.npy")].rsplit("_", 1)
        assert name != "Sample_Game_3"
        gt = bench.prepared(name, int(seed))[3]
        r = pm.evaluate(gt, np.load(f).astype(np.int8), bootstrap_n=0)
        rows[(name, int(seed))] = dict(spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"], poss=r["n_pred_spells"], gt=r["n_gt_spells"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    ref = score_dir(a.ref) if a.ref else None
    for d in a.dirs:
        rows = score_dir(d)
        line = "{:40s} n {} spellF1 {:.4f} macroF1 {:.4f} poss {}/{}".format(
            d[-40:], len(rows), np.mean([r["spellF1"] for r in rows.values()]), np.mean([r["macroF1"] for r in rows.values()]),
            sum(r["poss"] for r in rows.values()), sum(r["gt"] for r in rows.values()))
        if ref:
            common = sorted(set(rows) & set(ref))
            for k in ("spellF1", "macroF1"):
                dd = np.array([rows[u][k] - ref[u][k] for u in common])
                line += "  {} {:+.4f} ± {:.4f} ({}/{})".format(k, dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), int((dd > 0).sum()), len(dd))
        print(line)


if __name__ == "__main__":
    main()
