#!/usr/bin/env python3
"""Check the attack direction of every training game (PFF + DFL).

    python -I candidates/nn_pff/check_signs.py

Training uses sign = tune_on_sample_data.attack_sign (home mean x over the first 50 graded frames
of half 1 < 52.5 -> +1, the bench's kick-off rule) and features.frame_sign (flipped in half 2).
Independent check per half: each team's goalkeeper (the regular player whose mean x over the half
is furthest from halfway) must stand on its own side: home GK x < 52.5 in half 1 iff sign = +1,
the away GK on the other side, and both swap in half 2. Exit code 1 if any game fails. The old
half-mean rule (learned_dfl_evidence.make_dataset.attack_sign) is printed for reference only.
"""
import glob
import os
import sys

import numpy as np

WORK = os.environ["MTAH_WORK"]


def gk_x(z, half, team):
    m = z["half"] == half
    P = z["players"][m][:, z["team_of_player"] == team, 0]
    cnt = (~np.isnan(P)).sum(0)
    with np.errstate(all="ignore"):
        mx = np.nanmean(P, 0)
    dev = np.where(cnt > 0.5 * m.sum(), np.abs(mx - 52.5), -1)
    return float(mx[np.argmax(dev)])


def kick_sign(z):
    s = np.where(z["half"] == 1)[0][0]
    return 1.0 if np.nanmean(z["players"][s:s + 50, z["team_of_player"] == 0, 0]) < 52.5 else -1.0


def main():
    files = sorted(glob.glob(os.path.join(WORK, "pff_wc2022", "npz", "PFF-*.npz"))) + \
        sorted(glob.glob(os.path.join(WORK, "idsse", "npz", "DFL-*.npz")))
    bad = 0
    for f in files:
        z = np.load(f)
        s = kick_sign(z)
        h1, a1, h2, a2 = gk_x(z, 1, 0), gk_x(z, 1, 1), gk_x(z, 2, 0), gk_x(z, 2, 1)
        ok = ((h1 < 52.5) == (s > 0)) and ((a1 > 52.5) == (s > 0)) and \
             ((h2 > 52.5) == (s > 0)) and ((a2 < 52.5) == (s > 0))
        h = np.where(z["half"] == 1)[0]
        old = 1.0 if np.nanmean(z["players"][h][:, z["team_of_player"] == 0, 0]) < 52.5 else -1.0
        bad += not ok
        print("{:22s} sign {:+.0f}  GK x home h1 {:5.1f} h2 {:5.1f} away h1 {:5.1f} h2 {:5.1f}  {}{}".format(
            os.path.basename(f)[:-4], s, h1, h2, a1, a2, "ok" if ok else "FAIL",
            "" if old == s else "  (old half-mean rule would give {:+.0f})".format(old)))
    print("games {} failures {}".format(len(files), bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
