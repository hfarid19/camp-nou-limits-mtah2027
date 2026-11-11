#!/usr/bin/env python3
""""10-minutes-out" windows of the SkillCorner A-League games (round-5 prereg addendum 2026-11-10 11:30).

    python -I candidates/skc/window10.py            # -> $MTAH_WORK/skc_aleague/window10.json

Per game (rows of the 25 fps npz, $MTAH_WORK/skc_aleague/npz): s = first row of the first half,
window = [s + 30 min, s + 40 min) (minutes 30-40 of the first half, by frame index), drop = the window plus a 30 s guard
band each side ([s + 29.5 min, s + 40.5 min)). A model trained with nn_pff/data.py spec `nowin` (opt-in
NN_PFF_SKC_CV=1) never sees the drop rows of any SkillCorner game; the window rows are its in-game test set.

Usable window (fixed rule): drop inside the first half, alive (status 1) on >= 25% of the window rows and tracking
present (video True) on >= 25%. If minute 30-40 is not usable, the window is moved in 1-minute steps, trying -1, +1,
-2, +2, ... minutes, and the first usable one is taken (logged as "moved"). On the 20 games of 2026-11-10 every
minute-30-40 window was usable.
"""
import glob
import json
import os

import numpy as np

WORK = os.environ["MTAH_WORK"]
NPZ = os.path.join(WORK, "skc_aleague", "npz")
OUT = os.path.join(WORK, "skc_aleague", "window10.json")
MIN = 1500          # rows per minute at 25 fps
GUARD = 750         # 30 s


def usable(z, a, b, e):
    if a - GUARD < 0 or b + GUARD - 1 > e:
        return False
    return z["status"][a:b].mean() >= 0.25 and z["video"][a:b].mean() >= 0.25


def window(z):
    r = np.where(z["half"] == 1)[0]
    s, e = int(r[0]), int(r[-1])
    for k in [0] + [v for m in range(1, 30) for v in (-m, m)]:
        a = s + (30 + k) * MIN
        b = a + 10 * MIN
        if usable(z, a, b, e):
            return dict(win=[a, b], drop=[a - GUARD, b + GUARD], moved_min=k, half1=[s, e],
                        alive=float(z["status"][a:b].mean()), video=float(z["video"][a:b].mean()))
    raise RuntimeError("no usable window")


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(NPZ, "SKC-*.npz"))):
        g = os.path.basename(f)[:-4]
        out[g] = window(np.load(f))
        print(g, out[g], flush=True)
    json.dump(dict(rule=__doc__.strip().splitlines()[3:12], games=out), open(OUT, "w"), indent=1)
    print("moved:", [g for g, w in out.items() if w["moved_min"] != 0] or "none")


if __name__ == "__main__":
    main()
