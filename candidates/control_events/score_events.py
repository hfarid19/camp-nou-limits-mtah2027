#!/usr/bin/env python3
"""Score saved bench predictions against the event-defined controlled labels.

    python -I candidates/control_events/score_events.py <codes dir> [<codes dir> ...]
           [--labels DIR] [--per-game]

Reads <dir>/dfl2_<match>_<seed>_codes.npy (bench.py --dfl2 --save-probs) and
$MTAH_WORK/ctrl_events/<match>.npy (labels.py), and reports possession_metrics.evaluate's spellF1
and macroF1 against the event labels ("evt"), plus spellF1 against the raw DFL labels and the
ctrl_mean over control.fold T = 1, 2, 3 s for reference, and the predicted possession count.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import control                     # noqa: E402
import possession_metrics as pm    # noqa: E402

WORK = os.environ.get("MTAH_WORK", os.path.join(ROOT, "external", "work"))
LABELS = os.path.join(WORK, "ctrl_events")
NPZ = os.path.join(WORK, "idsse", "npz")
_cache = {}


def raw_gt(match):
    """bench.dfl2_prepared's gt, rebuilt from the npz without degrading anything."""
    if match not in _cache:
        z = np.load(os.path.join(NPZ, match + ".npz"))
        status, poss = z["status"], z["poss"]
        gt = np.full(len(status), pm.X, np.int8)
        gt[(status == 1) & (poss == 1)] = pm.H
        gt[(status == 1) & (poss == 2)] = pm.A
        gt[status == 0] = pm.D
        _cache[match] = gt
    return _cache[match]


def event_gt(match, labels=LABELS):
    return np.load(os.path.join(labels, match + ".npy"))


def score_dir(d, labels=LABELS, ctrl=True):
    """{(match, seed): dict(evt, evt_macro, raw, ctrl_mean, poss)}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "dfl2_*_codes.npy"))):
        match, seed = os.path.basename(f)[len("dfl2_"):-len("_codes.npy")].rsplit("_", 1)
        pred = np.load(f)
        ev = pm.evaluate(event_gt(match, labels), pred, bootstrap_n=0)
        r = dict(evt=ev["spell_f1_mean"], evt_macro=ev["macro_f1"], poss=ev["n_pred_spells"])
        if ctrl:
            gt = raw_gt(match)
            runs = pm.valid_runs(gt)
            r["raw"] = pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"]
            r["ctrl_mean"] = float(np.mean([pm.evaluate(control.fold(gt, T, runs), pred, bootstrap_n=0)
                                            ["spell_f1_mean"] for T in (1, 2, 3)]))
        out[(match, int(seed))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--per-game", action="store_true")
    args = ap.parse_args()
    print("{:24s} {:>3s} {:>8s} {:>8s} {:>8s} {:>9s} {:>6s}".format(
        "codes", "n", "evt", "evtMacro", "raw", "ctrl_mean", "poss"))
    for d in args.dirs:
        res = score_dir(d, args.labels)
        if not res:
            continue
        v = lambda k: np.mean([r[k] for r in res.values()])
        print("{:24s} {:3d} {:8.4f} {:8.4f} {:8.4f} {:9.4f} {:6d}".format(
            os.path.basename(d.rstrip("/")), len(res), v("evt"), v("evt_macro"), v("raw"), v("ctrl_mean"),
            int(np.sum([r["poss"] for r in res.values()]))))
        if args.per_game:
            for k, r in sorted(res.items()):
                print("   ", k, {a: round(b, 4) for a, b in r.items()})


if __name__ == "__main__":
    sys.exit(main())
