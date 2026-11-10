#!/usr/bin/env python3
"""Label-free statistics of pipeline_mbr.py on the BAR-ATH match vs the lsw rule on the SAME member outputs.

    python candidates/mbr_decode/match_run.py

Reads only the competition input, as make_submission.py does (ATD meta + raw tracking, Smart Tagging);
no ground truth, no scoring. Runs the three lsw members once, then (a) the lsw rule (= pipeline_lsw.py's
codes) and (b) the MBR decode (+ prekick_dead); prints possessions per 90, dead share, frames changed,
and the timing of each stage. Saves both code arrays to $MTAH_WORK/arch/mbr_decode/match_{lsw,mbr}_codes.npy.
"""

import os

for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import hashlib  # noqa: E402
import sys      # noqa: E402
import time     # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), HERE]
sys.dont_write_bytecode = True
import possession_metrics as pm   # noqa: E402
import possession_pipeline as pp  # noqa: E402
import pipeline_mbr as pmb        # noqa: E402
import decoder                    # noqa: E402
import v6_core                    # noqa: E402
import mbr                        # noqa: E402

OUT = os.path.join(os.environ.get("MTAH_WORK", ""), "arch", "mbr_decode")


def main():
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    args = (players, slot_team, ball, sign_half1, blocks, tags, names, GAMES[GAME]["pitch"], GAMES[GAME]["fps"])
    t0 = time.time()
    codes_list, probs = [], []
    for m in pmb._VOTE.members:
        t = time.time()
        codes_list.append(np.asarray(m.mod.predict(*args), np.int8))
        probs.append(np.asarray(m.mod.predict_proba(*args), np.float32).astype(np.float64))
        print("member {} {:.1f}s".format(m.slug, time.time() - t), flush=True)
    t1 = time.time()
    lsw = v6_core.prekick_dead(decoder.base(codes_list, blocks), tags, blocks)
    t = time.time()
    out, info = decoder.decode(probs, codes_list, blocks, pmb.CFG)
    out = v6_core.prekick_dead(np.asarray(out, np.int8), tags, blocks)
    t_dec = time.time() - t
    runs = mbr.runs_of(blocks)
    graded = np.zeros(n, bool)
    for a, b in runs:
        graded[a:b] = True
    min90 = graded.sum() / 25.0 / 60.0 / 90.0
    for name, c in (("lsw", lsw), ("mbr", out)):
        st = pm.possession_stats(c, runs)
        print("{:4s} possessions {} ({:.1f}/90), mean {:.1f} s, dead share {:.3f}, sha256 {}".format(
            name, st["n_spells"], st["n_spells"] / min90, st["mean_duration_s"], float((c[graded] == 2).mean()),
            hashlib.sha256(c.tobytes()).hexdigest()[:16]))
        if OUT:
            os.makedirs(OUT, exist_ok=True)
            np.save(os.path.join(OUT, "match_{}_codes.npy".format(name)), c)
    print("frames changed vs lsw: {:.2%} of graded ({} frames); windows {}, switched {}".format(
        float((out != lsw)[graded].mean()), int((out != lsw).sum()), info["n_windows"], info["n_switched"]))
    print("timing: members {:.1f}s, MBR decode {:.1f}s, total {:.1f}s for {} frames".format(
        t1 - t0, t_dec, time.time() - t0, n))
    t = time.time()
    again, _ = decoder.decode(probs, codes_list, blocks, pmb.CFG)
    again = v6_core.prekick_dead(np.asarray(again, np.int8), tags, blocks)
    print("decode repeat identical: {} ({:.1f}s)".format(bool(np.array_equal(again, out)), time.time() - t))


if __name__ == "__main__":
    main()
