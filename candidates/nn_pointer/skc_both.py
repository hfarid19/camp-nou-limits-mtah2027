#!/usr/bin/env python3
"""SkillCorner bench (20 games, full model) for one member in ONE pass per game: codes as `bench_skc.py run`
(probs/skc/<name>/skc2_<g>_2027_codes.npy) and predict_proba as `mbr_decode/skc_probs.py`
(probs/skc_mbr/<name>/skc2_<g>_2027.npy + .done + the codes copy). Same functions, same inputs; the pipeline's
evidence cache makes predict + predict_proba one forward pass. Games already done are skipped, so several
processes can split the list:  NN_PFF_MODEL=... python -I skc_both.py <pipeline.py> <name> [--games g ...]
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "skc")]
import bench_skc as bs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("name")
    ap.add_argument("--games", nargs="*", default=None)
    a = ap.parse_args()
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS"):
        assert k not in os.environ, k
    import tune_on_sample_data as tsd
    oc = os.path.join(bs.PROBS, a.name)
    om = os.path.join(bs.WORK, "probs", "skc_mbr", a.name)
    os.makedirs(oc, exist_ok=True)
    os.makedirs(om, exist_ok=True)
    module = None
    for g in (a.games or bs.all_games()):
        fc = os.path.join(oc, "skc2_{}_2027_codes.npy".format(g))
        fp = os.path.join(om, "skc2_{}_2027.npy".format(g))
        if os.path.isfile(fc) and os.path.isfile(fp + ".done"):
            continue
        if module is None:
            module = bs.load_candidate(a.pipeline, a.name)
        slots, slot_team, ball, gt, blocks, sign = bs.skc2_prepared(g)
        args = (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
        t0 = time.time()
        pred = np.asarray(module.predict(*args), np.int8)
        p = np.asarray(module.predict_proba(*args), np.float32)
        if not os.path.isfile(fc):
            np.save(fc, pred)
        np.save(fp, p)
        open(fp + ".done", "w").close()
        shutil.copyfile(fc, fp[:-4] + "_codes.npy")
        print(a.name, g, "{:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
