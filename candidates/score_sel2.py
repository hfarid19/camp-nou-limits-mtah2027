#!/usr/bin/env python3
"""Round-2 selection score (sel2) of saved DFL predictions, with paired comparisons.

    python candidates/score_sel2.py <codes dir> [<codes dir> ...] [--split dfl2|dfl3] [--ref DIR]

sel2 is the mean spellF1 over five readings of "controlled possession" on the 7 DFL games:
control.fold of the DFL flag at T = 1, 1.5 and 2 s, and the DFL event labels (control_events/)
with HOLD 3 s and HOLD 2 s. It was fixed on 2026-11-08, before any round-2 candidate was scored,
from two label-free pieces of evidence: Metrica's own event rules on its open sample games imply
305-331 possessions per 90 and T of about 1.2-1.5 s (control_events/metrica_rules.py), and the
DFL event labels imply 262-297 per 90 and T of about 2 s. The five targets span 262-344 per 90.
ctrl_mean (T = 1, 2, 3 s), the week-1 criterion, is printed alongside.

fine4 (fixed 2026-11-09, before any candidate was scored on it) is the mean over the raw DFL flag,
fold T = 0.5 and 1 s, and the event labels with HOLD 1 s. EVALUATION.md publishes the full-match
spellF1 of a perfect submission shifted 3 / 8 / 25 frames late (0.982 / 0.893 / 0.654); that
fingerprint depends only on the truth's spell lengths and lies between the raw DFL flag and a 1 s
fold, far from the 2 s fold (0.999 / 0.989 / 0.797). Metrica's practice labels match it
(0.94 / 0.88 / 0.66). fine4 replaces sel2 as the selection metric from 2026-11-09.

With --ref, every directory is also compared with the reference on the same game and seed:
difference of sel2, SE over the 7 game means, and game-seed wins.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench                     # noqa: E402
import possession_metrics as pm  # noqa: E402

EVT = {"evt3": os.path.join(bench.WORK, "ctrl_events"),
       "evt2": os.path.join(bench.WORK, "arch", "control_events", "variants", "hold2"),
       "evt1": os.path.join(bench.WORK, "arch", "control_events", "variants", "hold1")}
SEL2 = ("T1", "T1.5", "T2", "evt3", "evt2")
FINE = ("raw", "T0.5", "T1", "evt1")
CTRL = ("T1", "T2", "T3")


def targets(match, seed, split):
    gt = bench.dfl2_prepared(match, seed, 3 if split == "dfl3" else 2)[3]
    out = {"raw": gt}
    for T in (0.5, 1, 1.5, 2, 3):
        out["T{:g}".format(T)] = bench.controlled_labels(gt, T)
    for k, d in EVT.items():
        out[k] = np.load(os.path.join(d, match + ".npy"))
    return out


def score_dir(d, split):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, split + "_*_codes.npy"))):
        match, seed = os.path.basename(f)[len(split) + 1:-len("_codes.npy")].rsplit("_", 1)
        pred = np.load(f)
        r = {k: pm.evaluate(g, pred, bootstrap_n=0)["spell_f1_mean"]
             for k, g in targets(match, int(seed), split).items()}
        r["sel2"] = float(np.mean([r[k] for k in SEL2]))
        r["ctrl_mean"] = float(np.mean([r[k] for k in CTRL]))
        r["fine4"] = float(np.mean([r[k] for k in FINE]))
        r["poss"] = pm.evaluate(targets(match, int(seed), split)["raw"], pred, bootstrap_n=0)["n_pred_spells"]
        rows[(match, int(seed))] = r
    return rows


def paired(a, b, key="sel2"):
    common = sorted(set(a) & set(b))
    diff = {k: a[k][key] - b[k][key] for k in common}
    games = sorted({m for m, _ in common})
    gm = np.array([np.mean([v for (m, _), v in diff.items() if m == g]) for g in games])
    se = gm.std(ddof=1) / np.sqrt(len(gm)) if len(gm) > 1 else float("nan")
    wins = sum(v > 0 for v in diff.values())
    return float(gm.mean()), float(se), wins, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--split", default="dfl2")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()
    cols = ("raw", "T0.5", "evt1") + SEL2 + ("T3", "ctrl_mean", "sel2", "fine4")
    print("{:30s} {:>3s} ".format("codes", "n") + " ".join("{:>9s}".format(c) for c in cols) + "   poss")
    ref = score_dir(args.ref, args.split) if args.ref else None
    for d in args.dirs:
        rows = score_dir(d, args.split)
        m = {c: np.mean([r[c] for r in rows.values()]) for c in cols}
        line = "{:30s} {:3d} ".format(os.path.basename(d.rstrip("/"))[:30], len(rows))
        line += " ".join("{:9.4f}".format(m[c]) for c in cols)
        line += " {:6d}".format(int(np.sum([r["poss"] for r in rows.values()])))
        if ref is not None and d != args.ref:
            for key in ("fine4", "sel2"):
                mu, se, w, n = paired(rows, ref, key)
                line += "   {} {:+.4f} ± {:.4f} ({}/{})".format(key, mu, se, w, n)
        print(line)


if __name__ == "__main__":
    main()
