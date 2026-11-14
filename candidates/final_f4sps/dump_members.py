"""Bench candidate for the live = offline check (live_vs_offline.py --members-dir): runs pipeline.py unchanged (predict /
predict_proba are its own) and after each predict saves the outputs the pipeline itself used for that unit (pipeline's
_VOTE.last: the four members' probabilities, float32, and codes, then the stacker's as m4) to
$F4SPS_DUMP/<prefix>_<k>_m<i>[_codes].npy, where prefix = $F4SPS_PREFIX if set, else $BENCH_DFL_ONLY (one DFL game), else
"dev" (Metrica dev), k = the call index in bench order (DFL: seeds 2027, 2028, 2029; dev: Sample_Game_1 2027, 2028,
Sample_Game_2 2027, 2028; SkillCorner via skc_both.py: one call per game, so use one process per game with
F4SPS_PREFIX=skc_<game>) and i = the slot (0-3 = pipeline.MEMBERS, 4 = the stacker).

    BENCH_DFL_ONLY=DFL-MAT-J03WN1 F4SPS_DUMP=<dir> python -I candidates/bench.py candidates/final_f4sps/dump_members.py \
        --dfl2 --folds <folds json> --save-probs $MTAH_WORK/probs/r5_f4sps_live_dfl
    F4SPS_DUMP=<dir> python -I candidates/bench.py candidates/final_f4sps/dump_members.py --save-probs $MTAH_WORK/probs/r5_f4sps_live_met
    F4SPS_PREFIX=skc_<game> F4SPS_DUMP=<dir> python -I candidates/nn_pointer/skc_both.py \
        candidates/final_f4sps/dump_members.py r5_f4sps_live_skc --games <game>
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("f4sps_dump_base", os.path.join(HERE, "pipeline.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
OUT = os.environ["F4SPS_DUMP"]
PREFIX = os.environ.get("F4SPS_PREFIX") or os.environ.get("BENCH_DFL_ONLY") or "dev"
_N = [0]


def predict(*a):
    codes = base.predict(*a)
    os.makedirs(OUT, exist_ok=True)
    cl, pl = base._VOTE.last
    for i, (c, p) in enumerate(zip(cl, pl)):
        np.save(os.path.join(OUT, "{}_{}_m{}.npy".format(PREFIX, _N[0], i)), np.asarray(p, np.float32))
        np.save(os.path.join(OUT, "{}_{}_m{}_codes.npy".format(PREFIX, _N[0], i)), np.asarray(c, np.int8))
    _N[0] += 1
    return codes


predict_proba = base.predict_proba
