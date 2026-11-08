#!/usr/bin/env python3
"""Extract the ball FrameSets' (N, T) pairs from the raw DFL positions XMLs.

    python -I candidates/control_events/frametimes.py

Writes $MTAH_WORK/idsse/events/<match>_frametimes.npz with N (int32 DFL frame number) and
t (float64 UTC epoch seconds of the frame's T attribute), the clock the event XMLs' EventTime is
compared with. Read line by line with regexes; the XMLs are data only.
"""

import os
import re
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("MTAH_WORK", os.path.join(ROOT, "external", "work"))
RAW = os.path.join(WORK, "idsse", "raw")
OUT = os.path.join(WORK, "idsse", "events")
RE_N = re.compile(r'N="(\d+)" T="([^"]+)"')


def main():
    for f in sorted(os.listdir(RAW)):
        if "positions" not in f:
            continue
        match = re.search(r'(DFL-MAT-\w+)', f).group(1)
        out = os.path.join(OUT, match + "_frametimes.npz")
        if os.path.isfile(out):
            continue
        Ns, Ts, inball = [], [], False
        with open(os.path.join(RAW, f), encoding="utf-8") as h:
            for line in h:
                if "<FrameSet" in line:
                    inball = 'TeamId="BALL"' in line
                elif inball and "<Frame " in line:
                    m = RE_N.search(line)
                    Ns.append(int(m.group(1)))
                    Ts.append(datetime.fromisoformat(m.group(2)).timestamp())
        N, t = np.asarray(Ns, np.int32), np.asarray(Ts)
        np.savez(out, N=N, t=t)
        print(match, len(N), "dt values", np.unique(np.round(np.diff(t), 3), return_counts=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
