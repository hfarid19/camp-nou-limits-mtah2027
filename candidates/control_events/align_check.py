#!/usr/bin/env python3
"""Check how DFL event times line up with the tracking frames (README "Alignment").

    python -I candidates/control_events/align_check.py

For each game, at three stages (raw EventTime row; "frame" = positions clock after
events.align; "fframe" = flag clock after events.flag_lag):
  near     share of open-play player events whose acting player is within 1.5 m of the ball
           (positions only)
  restart  set-piece events (incl. kick-offs) vs the nearest BallStatus 0 -> 1 switch: median dt
           (s) and share within 0.5 s / 1 s. The flags are set live by an operator, so on the
           positions clock a positive median is the operator's lag.
  calc     CalculatedFrame (a frame DFL computed from tracking for shots) minus the event row
  flag     share of open-play controlled actions whose team the BallPossession flag names
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import events as E  # noqa: E402

FPS = 25


def stats(evs, z, key, pid):
    status, poss, P, B = z["status"], z["poss"], z["players"], z["ball"]
    on = E.restarts(status, z["half"])
    sp, calc, agree, near = [], [], [], []
    for e in evs:
        r = e[key]
        if r < 0:
            continue
        if e["restart"] >= 0:
            sp.append((on[np.argmin(np.abs(on - r))] - r) / FPS)
        if e["calc_row"] >= 0:
            calc.append((e["calc_row"] - r) / FPS)
        if e["ctrl"] and e["restart"] < 0 and status[r] == 1:
            agree.append(poss[r] - 1 == e["ctrl"][0][0])
        if e["restart"] < 0 and e["player"] in pid and e["kind"] in E.GEOM_KINDS:
            d = np.linalg.norm(P[r, pid[e["player"]]] - B[r])
            if np.isfinite(d):
                near.append(d < 1.5)
    return [np.array(v, float) for v in (near, sp, calc, agree)]


def fmt(near, sp, calc, agree):
    return ("near {:.2f} | restart dt med {:+.2f} <=0.5s {:.2f} <=1s {:.2f} | calc dt med {:+.2f} "
            "<=0.5s {:.2f} | flag {:.3f}").format(
        near.mean(), np.median(sp), np.mean(np.abs(sp) <= .5), np.mean(np.abs(sp) <= 1),
        np.median(calc), np.mean(np.abs(calc) <= .5), agree.mean())


def main():
    matches = sorted(f[:-4] for f in os.listdir(E.NPZ_DIR) if f.endswith(".npz"))
    keys = (("row", "raw    "), ("frame", "aligned"), ("fframe", "flagclk"))
    tot = {k: [[], [], [], []] for k, _ in keys}
    for match in matches:
        evs, z, info = E.load_aligned(match)
        pid = E.persons(match)
        print("{}  {} events | event clock offset {:+.2f} s, player events snapped {:.2f}, "
              "flag lag {:+.2f} s".format(match, len(evs), info["offset_s"], info["snapped"], info["flag_lag_s"]))
        for key, name in keys:
            s = stats(evs, z, key, pid)
            print("   ", name, fmt(*s))
            for i in range(4):
                tot[key][i].append(s[i])
    for key, name in keys:
        print("ALL", name, fmt(*[np.concatenate(v) for v in tot[key]]))


if __name__ == "__main__":
    main()
