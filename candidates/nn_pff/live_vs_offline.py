#!/usr/bin/env python3
"""Check that a live v6 run reproduces the offline vote of its members' saved codes, frame for frame.

    python candidates/nn_pff/live_vs_offline.py LIVE_NAME SMOOTH_S CONTROL_S MEMBER [MEMBER ...]

LIVE_NAME: probs/<LIVE_NAME> (bench --dfl2 codes) and probs/nn_pff/<LIVE_NAME> (run_pff.py codes);
MEMBER: "dfl_dir|pff_dir" as vote_offline.py. Prints the number of identical game(-seed)s.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote_offline as vo  # noqa: E402

name, sm, T, members = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
for dom, info in (("dfl", vo.dfl_info()), ("pff", vo.pff_info())):
    same = n = 0
    for u, (blocks, _) in info.items():
        f = (os.path.join(vo.WORK, "probs", name, "dfl2_{}_{}_codes.npy".format(*u)) if dom == "dfl" else
             os.path.join(vo.WORK, "probs", "nn_pff", name, "pff2_{}_2027_codes.npy".format(u)))
        live = np.load(f)
        off = vo.vote([vo.member_codes(m, dom, u) for m in members], blocks, sm, T)
        n += 1
        same += bool(np.array_equal(live, off))
    print("{}: live == offline on {}/{}".format(dom, same, n))
