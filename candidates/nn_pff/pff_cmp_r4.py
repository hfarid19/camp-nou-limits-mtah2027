#!/usr/bin/env python3
"""Round 4 PFF-only member screen (refs that have no DFL codes yet, e.g. the B ladder):
    python -I candidates/nn_pff/pff_cmp_r4.py CAND REF [REF ...]  -> paired PFF out-of-fold fine3 (members' own codes)."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import vote_r4 as vr  # noqa: E402

vo = vr.vo
c = sys.argv[1]
S = vr.score_set([c], (0.0, 0.0), ("pff",))
for r in sys.argv[2:]:
    R = vr.score_set([r], (0.0, 0.0), ("pff",))
    d, se, w, n = vo.paired_pff(S["pff"], R["pff"], "fine3")
    print("{:14s} vs {:14s} PFF fine3 {:.4f} vs {:.4f}: {:+.4f} ± {:.4f} ({}/{})".format(
        c, r, sum(x["fine3"] for x in S["pff"].values()) / n, sum(x["fine3"] for x in R["pff"].values()) / n, d, se, w, n))
