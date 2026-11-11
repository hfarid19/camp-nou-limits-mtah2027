#!/usr/bin/env python3
"""Live F4sP-MF2 codes (bench runs of pipeline.py, or of dump_members.py, which wraps it unchanged) == the offline decode
of the members' saved bench probabilities with pipeline.CFG? (prekick_dead is a no-op on the benches.)

    python -I candidates/final_f4spm/live_vs_offline.py --dfl-dir r5_f4spm_live_dfl --met-dir r5_f4spm_live_met \
        --skc-name r5_f4spm_live_skc [--members-dir DIR] [--mstcn-full NAME]

--dfl-dir / --met-dir: folders under $MTAH_WORK/probs written by bench.py --save-probs (DFL --dfl2 with the F4sP-MF2 folds
JSON; Metrica dev). --skc-name: a skc_both.py run (probs/skc/<name>). --members-dir: dump_members.py's per-member live
outputs; for each unit it also reports, per member, the largest |live - saved| probability and the number of frames
whose member codes differ.

Saved MS-TCN outputs. DFL: m5c_C192ls_skcft_metft, the leave-one-game-out fold models (the ones the scored F4sP-MF2 used;
the live bench resolves the shipped weights to the same fold models through the folds JSON). Metrica and SkillCorner:
the scored F4sP-MF2 used fold models there too (Metrica leave one game out, SkillCorner cross-validated), while the
pipeline loads the full model, which trained on all three Metrica games and all 20 SkillCorner games; so for these two
benches the MS-TCN slot is compared with --mstcn-full (default m5c_C192ls_skcft_metft_full): the full model's outputs
saved by the experiment's own scoring path (nn_pff/pipeline_cfg.py with NN_PFF_MODEL=<full.npz>, run_mac.sh "met" /
skc_both.py), not by the pipeline under test. The other three members' saved outputs are F4sP's (full models).
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
_spec = importlib.util.spec_from_file_location("final_f4spm_pipeline", os.path.join(HERE, "pipeline.py"))
f4spm = importlib.util.module_from_spec(_spec)   # unique name: members.load pops plain "pipeline" modules
_spec.loader.exec_module(f4spm)
OFF = ["adapt32/o_ls05__tta4_all_a50", "m5c_C192ls_skcft_metft", "radar_pff", "r5_ptr_ballc_p20M"]
DFL2_SEEDS = [2027, 2028, 2029]
DEV = [("Sample_Game_1", 2027), ("Sample_Game_1", 2028), ("Sample_Game_2", 2027), ("Sample_Game_2", 2028)]


def offline(domain, u, off):
    blocks, _ = data.info(domain, u)
    ms = [data.member(domain, m, u) for m in off]
    c, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p, _ in ms], [c for _, c in ms], blocks, f4spm.CFG)
    return np.asarray(c, np.int8), ms, blocks


def report(label, live, domain, u, off, dump=None):
    off_codes, ms, blocks = offline(domain, u, off)
    line = "{} {} {}".format(label, u, "identical" if np.array_equal(live, off_codes) else
                             "DIFFER in {} of {} frames".format(int((live != off_codes).sum()), len(off_codes)))
    if dump is not None and os.path.isfile(dump + "_m0.npy"):
        parts = []
        lp, lc = [], []
        for i, (p, c) in enumerate(ms):
            q, k = np.load(dump + "_m{}.npy".format(i)), np.load(dump + "_m{}_codes.npy".format(i))
            lp.append(q)
            lc.append(k)
            parts.append("m{} max|dP| {:.1e} codes differ {}".format(i, float(np.abs(q.astype(np.float64) - np.asarray(p, np.float64)).max()),
                                                                   int((k != np.asarray(c)).sum())))
        c2, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p in lp], lc, blocks, f4spm.CFG)
        parts.append("decode of the live member outputs == live codes: {}".format(np.array_equal(np.asarray(c2, np.int8), live)))
        line += " | " + "; ".join(parts)
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl-dir")
    ap.add_argument("--met-dir")
    ap.add_argument("--skc-name")
    ap.add_argument("--members-dir")
    ap.add_argument("--mstcn-full", default="m5c_C192ls_skcft_metft_full")
    a = ap.parse_args()
    full = OFF[:1] + [a.mstcn_full] + OFF[2:]
    print("offline members: DFL {}; Metrica / SkillCorner {}; CFG {}".format(OFF, full, f4spm.CFG), flush=True)
    P = os.path.join(data.WORK, "probs")
    if a.dfl_dir:
        for f in sorted(os.listdir(os.path.join(P, a.dfl_dir))):
            if f.endswith("_codes.npy"):
                m, s = f[len("dfl2_"):-len("_codes.npy")].rsplit("_", 1)
                live = np.load(os.path.join(P, a.dfl_dir, f))
                dump = os.path.join(a.members_dir, "{}_{}".format(m, DFL2_SEEDS.index(int(s)))) if a.members_dir else None
                report("DFL", live, "dfl", (m, int(s)), OFF, dump)
    if a.met_dir:
        for f in sorted(os.listdir(os.path.join(P, a.met_dir))):
            if f.endswith("_codes.npy"):
                g, s = f[len("dev_"):-len("_codes.npy")].rsplit("_", 1)
                live = np.load(os.path.join(P, a.met_dir, f))
                dump = os.path.join(a.members_dir, "dev_{}".format(DEV.index((g, int(s))))) if a.members_dir else None
                report("MET", live, "met", (g, int(s)), full, dump)
    if a.skc_name:
        for f in sorted(os.listdir(os.path.join(P, "skc", a.skc_name))):
            if f.endswith("_codes.npy"):
                g = f.split("_")[1]
                dump = os.path.join(a.members_dir, "skc_{}_0".format(g)) if a.members_dir else None
                report("SKC", np.load(os.path.join(P, "skc", a.skc_name, f)), "skc", g, full, dump)


if __name__ == "__main__":
    main()
