#!/usr/bin/env python3
"""Save the members' predict_proba on the 20 SkillCorner bench games (full models, as bench_skc run).

    NN_PFF_MODEL=... python candidates/mbr_decode/skc_probs.py <pipeline.py> <member name>

Writes $MTAH_WORK/probs/skc_mbr/<member>/skc2_<game>_2027.npy (float32 (N, 3)) and copies the
member's existing codes (probs/skc/<member>/..._codes.npy, written by bench_skc run) beside them.
One member per process. Existing files are skipped.
"""
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "skc")]
import bench_skc as bs  # noqa: E402


def main():
    path, name = sys.argv[1], sys.argv[2]
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS"):
        if k in os.environ:
            raise SystemExit(k + " is set")
    import tune_on_sample_data as tsd
    out = os.path.join(bs.WORK, "probs", "skc_mbr", name)
    os.makedirs(out, exist_ok=True)
    module = None
    for g in bs.all_games():
        f = os.path.join(out, "skc2_{}_2027.npy".format(g))
        src = os.path.join(bs.PROBS, name, "skc2_{}_2027_codes.npy".format(g))
        if not os.path.isfile(f + ".done"):
            if module is None:
                module = bs.load_candidate(path, name)
            slots, slot_team, ball, gt, blocks, sign = bs.skc2_prepared(g)
            args = (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
            t0 = time.time()
            p = np.asarray(module.predict_proba(*args), np.float32)
            np.save(f, p)
            open(f + ".done", "w").close()
            print(name, g, p.shape, "{:.1f}s".format(time.time() - t0), flush=True)
        shutil.copyfile(src, f[:-4] + "_codes.npy")


if __name__ == "__main__":
    main()
