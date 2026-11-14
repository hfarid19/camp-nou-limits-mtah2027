#!/usr/bin/env python3
"""Live v10 codes (bench runs of dump_members.py, which wraps pipeline.py unchanged) == the experiment's offline S2a decode
of saved outputs? (prekick_dead is a no-op on the benches.)

    python -I candidates/final_f4sps/live_vs_offline.py --dfl-dir r5_f4sps_live_dfl --met-dir r5_f4sps_live_met \
        --skc-name r5_f4sps_live_skc --members-dir DIR

Offline reference = arch/r5_arch/stack_aug/code/score_S.py's S2a decode: decoder.decode(four members + stacker, pipeline.CFG)
on the experiment's saved outputs:
  - DFL (--dfl2 units) and SkillCorner: the four F4sP members' saved bench outputs (DFL: leave-one-game-out fold models,
    which the live bench also resolves through the folds JSON) + the stacker trained on all 3 Metrica games
    ($MTAH_WORK/probs/stack_aug/S2_full), as scored.
  - Metrica: the members' outputs the experiment trained and scored on ($MTAH_WORK/probs/<dir>/met_<game>_<seed>, dirs
    adapt32/o_ls05_ref__tta4_all_a50, met_ft/parent_m5c_skcft, met_ft/parent_radar_pff, met_ft/parent_p20M_pod) + the
    ALL-GAMES stacker's in-sample outputs (S2_full/insample_met_*), NOT the leave-one-game-out stacker the scored S2a
    used on Metrica, because the shipped stacker is the all-games one.
--members-dir (dump_members.py's live outputs): per slot (m0-m3 members, m4 stacker) the largest |live - saved| probability
and the number of frames whose codes differ; the decode of the live outputs vs the live codes; and the shipped stacker
applied to the SAVED four members vs the live stacker (how much of a stacker difference the members' differences cause).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "nn_pointer"), os.path.join(CAND, "stacker_met"), HERE]
import finals    # noqa: E402  (patches mbr_decode.data._path for adapt32/* and stack_aug/* members)
import stacker   # noqa: E402

data, decoder = finals.data, finals.decoder
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("final_f4sps_pipeline", os.path.join(HERE, "pipeline.py"))
f4sps = importlib.util.module_from_spec(_spec)   # unique name: members.load pops plain "pipeline" modules
_spec.loader.exec_module(f4sps)
OFF = ["adapt32/o_ls05__tta4_all_a50", "m5c_C192ls_skcft", "radar_pff", "r5_ptr_ballc_p20M"]
MET_DIRS = ["adapt32/o_ls05_ref__tta4_all_a50", "met_ft/parent_m5c_skcft", "met_ft/parent_radar_pff",
            "met_ft/parent_p20M_pod"]
STACK = "stack_aug/S2_full"
DFL2_SEEDS = [2027, 2028, 2029]
DEV = [("Sample_Game_1", 2027), ("Sample_Game_1", 2028), ("Sample_Game_2", 2027), ("Sample_Game_2", 2028)]
BOOSTER = stacker.load()


def saved(domain, u):
    """-> blocks, [(probs float64, codes)] x 5 (four members, then the all-games stacker)."""
    blocks, _ = data.info(domain, u)
    if domain == "met":
        g, s = u
        ld = lambda d, f: (np.load(os.path.join(data.PROBS, d, f + ".npy")).astype(np.float64),
                           np.load(os.path.join(data.PROBS, d, f + "_codes.npy")).astype(np.int8))
        ms = [ld(d, "met_{}_{}".format(g, s)) for d in MET_DIRS] + [ld(STACK, "insample_met_{}_{}".format(g, s))]
    else:
        ms = [data.member(domain, m, u) for m in OFF + [STACK]]
    return blocks, ms


def dec(probs, codes, blocks):
    c, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p in probs], list(codes), blocks, f4sps.CFG)
    return np.asarray(c, np.int8)


def report(label, live, domain, u, dump=None):
    blocks, ms = saved(domain, u)
    off_codes = dec([p for p, _ in ms], [c for _, c in ms], blocks)
    line = "{} {} {}".format(label, u, "identical" if np.array_equal(live, off_codes) else
                             "DIFFER in {} of {} frames".format(int((live != off_codes).sum()), len(off_codes)))
    if dump is not None and os.path.isfile(dump + "_m0.npy"):
        parts, lp, lc = [], [], []
        for i, (p, c) in enumerate(ms):
            q, k = np.load(dump + "_m{}.npy".format(i)), np.load(dump + "_m{}_codes.npy".format(i))
            lp.append(q)
            lc.append(k)
            parts.append("m{} max|dP| {:.1e} codes differ {}".format(i, float(np.abs(q.astype(np.float64) - p).max()),
                                                                   int((k != c).sum())))
        c2 = dec(lp, lc, blocks)
        parts.append("decode of the live outputs == live codes: {}".format(np.array_equal(c2, live)))
        sp, sc = stacker.predict(lp[:4], blocks, BOOSTER)
        parts.append("shipped stacker on the live members == live stacker: {}".format(
            np.array_equal(sp, lp[4]) and np.array_equal(sc, lc[4])))
        sp, sc = stacker.predict([p for p, _ in ms[:4]], blocks, BOOSTER)
        parts.append("shipped stacker on the saved members vs saved stacker: max|dP| {:.1e}, codes differ {}".format(
            float(np.abs(sp.astype(np.float64) - ms[4][0]).max()), int((sc != ms[4][1]).sum())))
        line += " | " + "; ".join(parts)
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl-dir")
    ap.add_argument("--met-dir")
    ap.add_argument("--skc-name")
    ap.add_argument("--members-dir")
    a = ap.parse_args()
    print("offline: DFL / SkillCorner members {} + {}; Metrica members {} + {}/insample_met_*; CFG {}".format(
        OFF, STACK, MET_DIRS, STACK, f4sps.CFG), flush=True)
    P = os.path.join(data.WORK, "probs")
    if a.dfl_dir:
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
                g = f.split("_")[1]
                dump = os.path.join(a.members_dir, "skc_{}_0".format(g)) if a.members_dir else None
                report("SKC", np.load(os.path.join(P, "skc", a.skc_name, f)), "skc", g, dump)


if __name__ == "__main__":
    main()
