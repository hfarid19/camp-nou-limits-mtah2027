#!/usr/bin/env python3
"""Nested DFL figure for MBR stacks: each DFL game is decoded with the MBR config that R5-1's nested selection
picked for that game (g1 grid, chosen on the lsw members of the OTHER six games: mbr_decode/report.nested_dfl
on res_g1_dfl_lsw.pkl), so no DFL game is scored by a decoder chosen on itself. Member sets as stack_mbr.py.

    python candidates/adapt/nested_mbr.py --name NAME --nn TCN_SRC,MS_SRC

Codes -> $MTAH_WORK/probs/adapt/NAME/dfl2_<m>_<s>_codes.npy (DFL only; SKC / Metrica use the full-DFL pick =
pipeline_mbr.CFG, i.e. stack_mbr.py's output). Prints the picks.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(CAND, "mbr_decode")]
import stack_mbr as sm   # noqa: E402
import report            # noqa: E402
import decoder           # noqa: E402

md = sm.md


def picks():
    R = pickle.load(open(os.path.join(md.WORK, "arch", "mbr_decode", "res_g1_dfl_lsw.pkl"), "rb"))
    _, p, full = report.nested_dfl(R)
    return {g: dict(k) for g, k in p.items()}, dict(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--nn", required=True)
    a = ap.parse_args()
    srcs = a.nn.split(",")
    P, full = picks()
    print("full-DFL pick", full)
    for g, c in sorted(P.items()):
        print("  ", g, c)
    out = os.path.join(md.WORK, "probs", "adapt", a.name)
    os.makedirs(out, exist_ok=True)
    for u in md.units("dfl"):
        blocks, _ = md.info("dfl", u)
        ms = [sm.nn_member(srcs[0], "o_ls05", "dfl", u), sm.nn_member(srcs[1], "m5c_W", "dfl", u),
              md.member("dfl", "radar_pff", u)]
        probs = [np.asarray(p, np.float32).astype(np.float64) for p, _ in ms]
        codes, _ = decoder.decode(probs, [c for _, c in ms], blocks, P[u[0]])
        np.save(os.path.join(out, "dfl2_{}_{}_codes.npy".format(*u)), np.asarray(codes, np.int8))


if __name__ == "__main__":
    main()
