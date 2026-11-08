#!/usr/bin/env python3
"""Offline replica of pipeline.py on saved member codes (bench --save-probs dirs), for choosing
which version of each member to vote with before the live run.

    python candidates/ensemble_ctrl_v2/vote_offline.py OUT_DIR MEMBER_DIR [MEMBER_DIR ...]

MEMBER_DIRs in tie-break order. For every dfl2_<match>_<seed>_codes.npy present in all of them it
writes OUT_DIR/dfl2_<match>_<seed>_codes.npy = control.fold(merge_short_v2(hard vote), 2.0 s),
the same operations as pipeline.py (ensemble_core.hard_vote + control.fold). Score the result
with candidates/score_codes.py or temporal_tcn_v2/paired_codes.py.
"""

import glob
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.insert(0, CAND)
sys.path.insert(0, HERE)
import bench                # noqa: E402
import control              # noqa: E402
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402

SMOOTH = dict(min_poss_s=1.0, min_dead_s=1.0)     # = pipeline.py
CONTROL_S = 2.0


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    lde = members.load("learned_dfl_evidence")
    names = None
    for d in dirs:
        s = {os.path.basename(f) for f in glob.glob(os.path.join(d, "dfl2_*_codes.npy"))}
        names = s if names is None else names & s
    for nm in sorted(names):
        match, seed = nm[len("dfl2_"):-len("_codes.npy")].rsplit("_", 1)
        blocks = bench.dfl2_prepared(match, int(seed))[4]
        blocks = [(int(h), int(f), int(l)) for h, f, l in blocks]
        codes_list = [np.load(os.path.join(d, nm)).astype(np.int8) for d in dirs]
        codes = ec.hard_vote(codes_list, blocks, lde, 25, None, SMOOTH)
        runs = [(f, l + 1) for _, f, l in blocks]
        codes = control.fold(np.asarray(codes, np.int8), CONTROL_S, runs, 25)
        np.save(os.path.join(out, nm), codes)
    print("wrote", len(names), "to", out)


if __name__ == "__main__":
    main()
