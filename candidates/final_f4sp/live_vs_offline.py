#!/usr/bin/env python3
"""Live F4sP codes (bench.py runs of pipeline.py, or of dump_members.py, which wraps it unchanged) == the offline decode
of the members' saved bench probabilities (candidates/nn_pointer/finals.py's inputs) with pipeline.CFG?
(prekick_dead is a no-op on the benches.)

    python -I candidates/final_f4sp/live_vs_offline.py --dfl-dir r5_f4sp_live_dfl --met-dir r5_f4sp_live_met \
        [--skc-name NAME] [--members-dir DIR]

--dfl-dir / --met-dir: folders under $MTAH_WORK/probs written by bench.py --save-probs (DFL --dfl2 with the F4sP folds
JSON; Metrica dev). --skc-name: a bench_skc.py run under probs/skc. Not used for F4sP: the saved SkillCorner
probabilities of m5c_C192ls_skcft are cross-validated (each game scored by the fold model that never saw it), while
the pipeline loads the full model, which trained on all 20 SkillCorner games, so they are expected to differ.
--members-dir: dump_members.py's per-member live outputs; for each unit it also reports, per member, the largest
|live - saved| probability and the number of frames whose member codes differ.
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
_spec = importlib.util.spec_from_file_location("final_f4sp_pipeline", os.path.join(HERE, "pipeline.py"))
f4sp = importlib.util.module_from_spec(_spec)   # unique name: members.load pops plain "pipeline" modules
_spec.loader.exec_module(f4sp)
OFF = ["adapt32/o_ls05__tta4_all_a50", "m5c_C192ls_skcft", "radar_pff", "r5_ptr_ballc_p20M"]
DFL2_SEEDS = [2027, 2028, 2029]
DEV = [("Sample_Game_1", 2027), ("Sample_Game_1", 2028), ("Sample_Game_2", 2027), ("Sample_Game_2", 2028)]


def offline(domain, u):
    blocks, _ = data.info(domain, u)
    ms = [data.member(domain, m, u) for m in OFF]
    c, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p, _ in ms], [c for _, c in ms], blocks, f4sp.CFG)
    return np.asarray(c, np.int8), ms, blocks


def report(label, live, domain, u, dump=None):
    off, ms, blocks = offline(domain, u)
    line = "{} {} {}".format(label, u, "identical" if np.array_equal(live, off) else
                             "DIFFER in {} of {} frames".format(int((live != off).sum()), len(off)))
    if dump is not None and os.path.isfile(dump + "_m0.npy"):
        parts = []
        lp, lc = [], []
        for i, (p, c) in enumerate(ms):
            q, k = np.load(dump + "_m{}.npy".format(i)), np.load(dump + "_m{}_codes.npy".format(i))
            lp.append(q)
            lc.append(k)
            parts.append("m{} max|dP| {:.1e} codes differ {}".format(i, float(np.abs(q.astype(np.float64) - np.asarray(p, np.float64)).max()),
                                                                   int((k != np.asarray(c)).sum())))
        c2, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p in lp], lc, blocks, f4sp.CFG)
        parts.append("decode of the live member outputs == live codes: {}".format(np.array_equal(np.asarray(c2, np.int8), live)))
        line += " | " + "; ".join(parts)
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl-dir")
    ap.add_argument("--met-dir")
    ap.add_argument("--skc-name")
    ap.add_argument("--members-dir")
    a = ap.parse_args()
    P = os.path.join(data.WORK, "probs")
    if a.dfl_dir:
        idx = {}
        for f in sorted(os.listdir(os.path.join(P, a.dfl_dir))):
            if f.endswith("_codes.npy"):
                m, s = f[len("dfl2_"):-len("_codes.npy")].rsplit("_", 1)
                live = np.load(os.path.join(P, a.dfl_dir, f))
                dump = os.path.join(a.members_dir, "{}_{}".format(m, DFL2_SEEDS.index(int(s)))) if a.members_dir else None
                report("DFL", live, "dfl", (m, int(s)), dump)
    if a.met_dir:
        for f in sorted(os.listdir(os.path.join(P, a.met_dir))):
            if f.endswith("_codes.npy"):
                g, s = f[len("dev_"):-len("_codes.npy")].rsplit("_", 1)
                live = np.load(os.path.join(P, a.met_dir, f))
                dump = os.path.join(a.members_dir, "dev_{}".format(DEV.index((g, int(s))))) if a.members_dir else None
                report("MET", live, "met", (g, int(s)), dump)
    if a.skc_name:
        for f in sorted(os.listdir(os.path.join(P, "skc", a.skc_name))):
            if f.endswith("_codes.npy"):
                report("SKC", np.load(os.path.join(P, "skc", a.skc_name, f)), "skc", f.split("_")[1])


if __name__ == "__main__":
    main()
