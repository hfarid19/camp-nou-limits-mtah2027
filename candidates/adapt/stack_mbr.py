#!/usr/bin/env python3
"""Re-normalisation stacked with R5-1's MBR decoder (fixed config pipeline_mbr.CFG, no re-tuning), offline.

    python candidates/adapt/stack_mbr.py --name NAME --nn TCN_DIR,MS_DIR [--shard i --nshard n]

Members: the two NN members' float32 probabilities + codes from TCN_DIR / MS_DIR (folders under
$MTAH_WORK/probs/adapt32 written by run_renorm.py --p32, or "orig" = mbr_decode/data.py's saved lsw
member outputs) and radar_pff's saved outputs (mbr_decode/data.py), in lsw order. decoder.decode as
pipeline_mbr.py (prekick_dead is a no-op on the benches). Codes -> $MTAH_WORK/probs/adapt/NAME/<tag>_codes.npy,
scored with score_adapt.py as "final:NAME".
"""

import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAND, "mbr_decode"))
import data as md     # noqa: E402  (mbr_decode/data.py)
import decoder        # noqa: E402
import pipeline_mbr   # noqa: E402

WORK = md.WORK
TAG = {"dfl": "dfl2_{}_{}", "skc": "skc2_{}_2027", "met": "dev_{}_{}"}


def nn_member(src, cfg, dom, u):
    if src == "orig":
        return md.member(dom, cfg, u)
    tag = TAG[dom].format(*(u if isinstance(u, tuple) else (u,)))
    f = os.path.join(WORK, "probs", "adapt32", src, tag)
    if not os.path.isfile(f + ".npy"):                      # float16 outputs (TTA): exploratory reads only
        f = os.path.join(WORK, "probs", "adapt", src, tag)
    return np.load(f + ".npy").astype(np.float32).astype(np.float64), np.load(f + "_codes.npy").astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--nn", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    srcs = a.nn.split(",")
    out = os.path.join(WORK, "probs", "adapt", a.name)
    os.makedirs(out, exist_ok=True)
    work = [(d, u) for d in ("dfl", "skc", "met") for u in md.units(d)][a.shard::a.nshard]
    for dom, u in work:
        tag = TAG[dom].format(*(u if isinstance(u, tuple) else (u,)))
        f = os.path.join(out, tag + "_codes.npy")
        if os.path.isfile(f):
            continue
        blocks, _ = md.info(dom, u)
        ms = [nn_member(srcs[0], "o_ls05", dom, u), nn_member(srcs[1], "m5c_W", dom, u),
              md.member(dom, "radar_pff", u)]
        probs = [np.asarray(p, np.float32).astype(np.float64) for p, _ in ms]
        codes, _ = decoder.decode(probs, [c for _, c in ms], blocks, pipeline_mbr.CFG)
        np.save(f, np.asarray(codes, np.int8))
        print(tag, flush=True)


if __name__ == "__main__":
    main()
