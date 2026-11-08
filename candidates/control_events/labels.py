#!/usr/bin/env python3
"""Event-defined "controlled possession" labels for the 7 open DFL games (README "Rules").

    python -I candidates/control_events/labels.py [--tol 0.5] [--hold 3.0] [--out DIR]
                                                   [--touch-counts] [--snapped-only]

Writes <out>/<match>.npy (int8 per raw DFL frame, possession_metrics codes H 0, A 1, D 2, X 3;
same rows as idsse/npz/<match>.npz and as gt in bench.dfl2_prepared). Default out:
$MTAH_WORK/ctrl_events. Also importable: event_labels(gt, evs, tol, hold).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import events as E                # noqa: E402
import possession_metrics as pm   # noqa: E402

FPS = 25
TOL_S = 0.5     # an action counts for a spell if it lies within the spell +- TOL_S (flag clock)
HOLD_S = 3.0    # a DFL spell this long counts as controlled without any event


def dfl_gt(z):
    """Exactly bench.dfl2_prepared's gt: D where BallStatus 0, H/A from BallPossession, else X."""
    status, poss = z["status"], z["poss"]
    gt = np.full(len(status), pm.X, np.int8)
    gt[(status == 1) & (poss == 1)] = pm.H
    gt[(status == 1) & (poss == 2)] = pm.A
    gt[status == 0] = pm.D
    return gt


def control_marks(evs, n, touch_counts=False, snapped_only=False):
    """(n, 2) bool: frame f, team k has a controlled action at f (on the flag clock)."""
    m = np.zeros((n, 2), bool)
    for e in evs:
        f = e.get("fframe", -1)
        if f < 0 or (snapped_only and not e.get("snapped")):
            continue
        acts = list(e["ctrl"])
        if touch_counts:
            acts += [t for t in e["touch"] if t[1] == "other_ball_action"]
        for team, _ in acts:
            if team in (0, 1):
                m[f, team] = True
    return m


def event_labels(gt, marks, tol=TOL_S, hold=HOLD_S, fps=FPS, return_spells=False):
    """Hand back every DFL spell that is not a controlled change (one left-to-right pass).

    A DFL spell of team X that directly follows possession of the other team Y (no stoppage in
    between) is a change of possession only if, within its frames +- tol s, X has a controlled
    action (marks), or it lasts at least hold s. Otherwise its frames get Y. Spells after a
    stoppage or at the start of a graded run are always kept. D and X frames are never changed.
    """
    out = gt.copy()
    t = int(round(tol * fps))
    spells = []
    owner = None
    prev_end = None
    for a, b, c in pm.segments(gt, pm.valid_runs(gt)):
        if c not in (pm.H, pm.A):
            owner = None
            continue
        if owner is None or prev_end != a:
            owner = c                      # after a stoppage / run start: clear possessor
        elif c != owner:
            ev = marks[max(0, a - t):b + t, c].any()
            long_ = (b - a) >= hold * fps
            keep = bool(ev or long_)
            spells.append((a, b, c, owner, keep, bool(ev), bool(long_)))
            if keep:
                owner = c
            else:
                out[a:b] = owner
        prev_end = b
    return (out, spells) if return_spells else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=TOL_S)
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--touch-counts", action="store_true", help="OtherBallAction counts as control")
    ap.add_argument("--snapped-only", action="store_true", help="only events placed by tracking")
    ap.add_argument("--out", default=os.path.join(E.WORK, "ctrl_events"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    matches = sorted(f[:-4] for f in os.listdir(E.NPZ_DIR) if f.endswith(".npz"))
    info = {}
    for match in matches:
        evs, z, al = E.load_aligned(match)
        gt = dfl_gt(z)
        marks = control_marks(evs, len(gt), args.touch_counts, args.snapped_only)
        lab, spells = event_labels(gt, marks, args.tol, args.hold, return_spells=True)
        np.save(os.path.join(args.out, match + ".npy"), lab.astype(np.int8))
        sp = np.array([s[4:] for s in spells], bool).reshape(-1, 3)
        info[match] = dict(al, n_contested=len(spells), kept=int(sp[:, 0].sum()),
                           kept_by_event=int(sp[:, 1].sum()), kept_by_hold_only=int((sp[:, 2] & ~sp[:, 1]).sum()),
                           raw_spells=sum(1 for s in pm.segments(gt, pm.valid_runs(gt)) if s[2] in (0, 1)),
                           event_spells=sum(1 for s in pm.segments(lab, pm.valid_runs(lab)) if s[2] in (0, 1)))
        print(match, info[match], flush=True)
    with open(os.path.join(args.out, "labels_info.json"), "w") as f:
        json.dump(dict(params=vars(args), games=info), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
