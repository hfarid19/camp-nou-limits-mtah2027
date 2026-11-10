#!/usr/bin/env python3
"""Live F4b codes (bench.py --dfl2 with the F4b folds JSON / bench_skc.py run) == the offline decode of the members' saved
probabilities (candidates/nn_pointer/finals.py's inputs) with pipeline.CFG?  (prekick_dead is a no-op on the benches.)
    python -I candidates/final_f4b/live_vs_offline.py --dfl-dir probs/r5_f4b_live_dfl --skc-name r5_f4b_live
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "nn_pointer"), HERE]
import finals    # noqa: E402  (patches mbr_decode.data._path for adapt32/* members)

data, decoder = finals.data, finals.decoder
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("final_f4b_pipeline", os.path.join(HERE, "pipeline.py"))
f4b = importlib.util.module_from_spec(_spec)   # unique name: members.load pops plain "pipeline" modules
_spec.loader.exec_module(f4b)
OFF = ["adapt32/o_ls05__tta4_all_a50", "m5c_C192ls", "radar_pff", "r5_ptr_ballc"]


def offline(domain, u):
    blocks, _ = data.info(domain, u)
    ms = [data.member(domain, m, u) for m in OFF]
    c, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p, _ in ms], [c for _, c in ms], blocks, f4b.CFG)
    return np.asarray(c, np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl-dir", required=True)
    ap.add_argument("--skc-name", required=True)
    a = ap.parse_args()
    P = os.path.join(data.WORK, "probs")
    for f in sorted(os.listdir(os.path.join(P, a.dfl_dir))):
        if f.endswith("_codes.npy"):
            m, s = f[len("dfl2_"):-len("_codes.npy")].rsplit("_", 1)
            live = np.load(os.path.join(P, a.dfl_dir, f))
            off = offline("dfl", (m, int(s)))
            print("DFL", m, s, "identical" if np.array_equal(live, off) else "DIFFER in {} frames".format(int((live != off).sum())))
    for f in sorted(os.listdir(os.path.join(P, "skc", a.skc_name))):
        if f.endswith("_codes.npy"):
            g = f.split("_")[1]
            live = np.load(os.path.join(P, "skc", a.skc_name, f))
            off = offline("skc", g)
            print("SKC", g, "identical" if np.array_equal(live, off) else "DIFFER in {} frames".format(int((live != off).sum())))


if __name__ == "__main__":
    main()
