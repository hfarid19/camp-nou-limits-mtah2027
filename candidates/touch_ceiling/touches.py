#!/usr/bin/env python3
"""Step 1: the true touch sequence of each selection game, and the check that it reproduces the label.

    python candidates/touch_ceiling/touches.py          # writes $MTAH_WORK/arch/touch_ceiling/touches.pkl

A touch = an on-the-ball game event (OTB / FIRSTKICKOFF / SECONDKICKOFF with home_team set) inside
the period's source frames, exactly the events labels.build(..., "touch") turns into team changes.
Its 25 fps row is the first graded row whose PFF frameNum (npz `frame`, the nearest source frame of
the sample) is >= the event's start_frame: the label of a row is the label of that source frame, so
the team switches there. Dead = npz status 0 (OUT -> next on-ball event).

Per game: rows of each touch, team (0 H / 1 A in pm codes), kind (0 restart = first touch of a
period or after a dead row, 1 change = team differs from the previous touch in open play, 2 same
team), alive mask, half boundaries. Check: rebuilding H/A/D from touches + dead gives spellF1 1.0
and an identical array versus the bench's touch target.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ON_BALL = ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF")


def extract(game):
    z = np.load(os.path.join(C.PFF, "npz", "PFF-" + game + ".npz"))
    src = np.load(os.path.join(C.PFF, "src", game + ".npz"))
    sf, sp = src["frame"], src["period"]
    ev = json.load(open(os.path.join(C.PFF, "src", game + "_events.json")))
    frame, half, status = z["frame"], z["half"], z["status"]
    rows, teams = [], []
    halves = []
    for per in (1, 2):
        idx = np.where(half == per)[0]
        halves.append((int(idx[0]), int(idx[-1]) + 1))
        f = sf[sp == per]
        lo, hi = int(f[0]), int(f[-1])
        evs = [e for e in ev["game_events"] if e["start_frame"] is not None and lo <= int(e["start_frame"]) <= hi
               and e["type"] in ON_BALL and e.get("home_team") is not None]
        evs.sort(key=lambda e: (int(e["start_frame"]), e["id"]))
        fr = frame[idx]
        for e in evs:
            k = int(np.searchsorted(fr, int(e["start_frame"]), side="left"))
            k = min(k, len(fr))                     # after the last graded row: never takes effect
            rows.append(int(idx[0]) + k if k < len(fr) else -1)
            teams.append(C.H if int(e["home_team"]) == 1 else C.A)
    rows, teams = np.array(rows), np.array(teams, np.int8)
    keep = rows >= 0
    rows, teams = rows[keep], teams[keep]
    alive = status == 1
    kind = np.full(len(rows), 2, np.int8)
    starts = {h0 for h0, _ in halves}
    for i, r in enumerate(rows):
        new_half = i == 0 or any(rows[i - 1] < h0 <= r for h0 in starts)
        if new_half or r in starts or not alive[r - 1] or not alive[rows[i - 1]:r + 1].all():
            kind[i] = 0                             # kick-off / restart after dead
        elif teams[i] != teams[i - 1]:
            kind[i] = 1                             # open-play team change (interception, tackle, ...)
    return dict(rows=rows, teams=teams, kind=kind, alive=alive, halves=halves, n=len(status),
                poss=z["poss"], status=status)


def build_labels(rows, teams, alive, halves, n, fallback=None):
    """H/A/D from touches: team = last touch at or before the row (within the half); rows before the
    first touch of a half take that touch's team (labels.py), or `fallback` if given; dead -> D."""
    out = np.empty(n, np.int8)
    for h0, h1 in halves:
        m = (rows >= h0) & (rows < h1)
        r, t = rows[m], teams[m]
        o = np.argsort(r, kind="stable")
        r, t = r[o], t[o]
        pos = np.searchsorted(r, np.arange(h0, h1), side="right") - 1
        if len(r):
            seg = np.where(pos >= 0, t[np.maximum(pos, 0)], t[0] if fallback is None else fallback[h0:h1])
        else:
            seg = fallback[h0:h1] if fallback is not None else np.full(h1 - h0, C.H, np.int8)
        out[h0:h1] = seg
    out[~alive] = C.D
    return out


def main():
    T = C.targets()
    data = {}
    worst = 1.0
    tot = np.zeros(3, int)
    for g in C.selection_games():
        d = extract(g)
        gt = T[g][1]["raw"]
        assert len(gt) == d["n"] and (gt != C.X).all()
        lab = build_labels(d["rows"], d["teams"], d["alive"], d["halves"], d["n"])
        same = bool((lab == gt).all())
        f1 = C.spell_f1(C.gt_spells(gt), lab)
        f1_ref = C.pm.spell_scores(gt, lab, C.pm.valid_runs(gt))["mean_f1"]
        worst = min(worst, f1)
        cnt = np.bincount(d["kind"], minlength=3)
        tot += cnt
        print("{:6s} touches {:5d} (restart {:4d} change {:4d} same {:5d})  identical {}  spellF1 {:.4f} (pm {:.4f})".format(
            g, len(d["rows"]), *cnt, same, f1, f1_ref), flush=True)
        assert same and f1 == 1.0 and f1_ref == 1.0
        data[g] = d
    os.makedirs(C.OUT, exist_ok=True)
    pickle.dump(data, open(os.path.join(C.OUT, "touches.pkl"), "wb"))
    print("48 games: every label rebuilt exactly (min spellF1 {:.4f}); touches restart {} change {} same {}".format(
        worst, *tot))


if __name__ == "__main__":
    main()
