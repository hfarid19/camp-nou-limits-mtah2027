#!/usr/bin/env python3
"""Label-free statistics of the re-normalised lsw members on the BAR-ATH match (Mac only; positions and tags only).

    python candidates/adapt/match_run.py

Reads only the competition input, as make_submission.py does; no ground truth, no scoring. Runs the lsw
members and the renorm members (all_a50, all_a100: renorm_members.py, i.e. the live member files the
pipelines ship), then for each member set (a) the v6 rule (= pipeline_lsw / pipeline_lsw_renorm_*) and
(b) the fixed MBR decode (mbr_decode/pipeline_mbr.CFG), + prekick_dead. Prints possessions per 90, dead
share, frames changed vs lsw and vs MBR-lsw, the sha256 of each output, and a determinism check (the
all_a100 members re-loaded and re-run must give byte-identical probabilities and codes). Codes are
saved to $MTAH_WORK/arch/r5_adapt/match_<name>_codes.npy (local only).
"""

import os

for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import hashlib  # noqa: E402
import sys      # noqa: E402
import time     # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(CAND, "ensemble_ctrl_v6"),
                os.path.join(CAND, "mbr_decode"), HERE]
import possession_metrics as pm   # noqa: E402
import possession_pipeline as pp  # noqa: E402
import members                    # noqa: E402
import v6_core                    # noqa: E402
import decoder                    # noqa: E402
import pipeline_mbr               # noqa: E402

OUT = os.path.join(os.environ["MTAH_WORK"], "arch", "r5_adapt")
SETS = {
    "lsw": [("nn_pff_tcn", "pipeline_ls.py"), ("nn_pff", "pipeline_w.py")],
    "a50": [("adapt", "member_tcn_a50.py"), ("adapt", "member_mstcn_a50.py")],
    "a100": [("adapt", "member_tcn_a100.py"), ("adapt", "member_mstcn_a100.py")],
    "Ttta4": [("adapt", "member_tcn_tta4.py"), ("nn_pff", "pipeline_w.py")],
    "Ttta4_a50": [("adapt", "member_tcn_tta4_a50.py"), ("nn_pff", "pipeline_w.py")],
}


def run_members(specs, args):
    out = []
    for s, f in specs:
        m = members.Member(s, f).mod
        out.append((np.asarray(m.predict(*args), np.int8), np.asarray(m.predict_proba(*args), np.float32)))
    return out


def main():
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    fps = GAMES[GAME]["fps"]
    args = (players, slot_team, ball, sign_half1, blocks, tags, names, GAMES[GAME]["pitch"], fps)
    t = time.time()
    radar = run_members([("ball_radar_pff", "pipeline.py")], args)[0]
    print("radar {:.0f}s".format(time.time() - t), flush=True)
    runs = [(int(f), int(l) + 1) for _, f, l in blocks]
    graded = np.zeros(n, bool)
    for a, b in runs:
        graded[a:b] = True
    min90 = graded.sum() / float(fps) / 60.0 / 90.0
    res, mem = {}, {}
    for name, specs in SETS.items():
        t = time.time()
        ms = run_members(specs, args) + [radar]
        mem[name] = ms
        cl = [c for c, _ in ms]
        probs = [p.astype(np.float64) for _, p in ms]
        res[name + "+rule"] = v6_core.prekick_dead(np.asarray(decoder.base(cl, blocks), np.int8), tags, blocks)
        mb, _ = decoder.decode(probs, cl, blocks, pipeline_mbr.CFG)
        res[name + "+mbr"] = v6_core.prekick_dead(np.asarray(mb, np.int8), tags, blocks)
        print("{} members {:.0f}s; member frames changed vs lsw: {}".format(
            name, time.time() - t, ", ".join("{:.2%}".format(float((c != c0)[graded].mean()))
                                             for (c, _), (c0, _) in zip(ms, mem["lsw"]))), flush=True)
    for k, c in res.items():
        st = pm.possession_stats(c, runs)
        print("{:9s} possessions {} ({:.1f}/90), dead share {:.3f}, changed vs lsw {:.2%}, vs MBR-lsw {:.2%}, sha256 {}".format(
            k, st["n_spells"], st["n_spells"] / min90, float((c[graded] == 2).mean()),
            float((c != res["lsw+rule"])[graded].mean()), float((c != res["lsw+mbr"])[graded].mean()),
            hashlib.sha256(c.tobytes()).hexdigest()[:16]), flush=True)
        np.save(os.path.join(OUT, "match_{}_codes.npy".format(k.replace("+", "_"))), c)
    for k in ("a100", "Ttta4_a50"):
        again = run_members(SETS[k], args)
        same = all(np.array_equal(c, c2) and np.array_equal(p, p2) for (c, p), (c2, p2) in zip(again, mem[k][:2]))
        print("determinism: {} members re-loaded and re-run -> identical codes and probs: {}".format(k, same))


if __name__ == "__main__":
    main()
