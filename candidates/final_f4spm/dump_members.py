"""Bench candidate for the live = offline check (live_vs_offline.py --members-dir): runs pipeline.py unchanged (predict /
predict_proba are its own) and after each predict also saves every member's live probabilities (float32) and codes to
$F4SPM_DUMP/<prefix>_<k>_m<i>[_codes].npy, where prefix = $F4SPM_PREFIX if set, else $BENCH_DFL_ONLY (one DFL game),
else "dev" (Metrica dev), k = the call index in bench order (DFL: seeds 2027, 2028, 2029; dev: Sample_Game_1 2027, 2028,
Sample_Game_2 2027, 2028; SkillCorner via skc_both.py: one call per game, so use one process per game with
F4SPM_PREFIX=skc_<game>) and i = the member's slot in pipeline.MEMBERS. The pointer and MS-TCN members cache their
outputs, so only the TCN and radar members are recomputed for the dump.

    BENCH_DFL_ONLY=DFL-MAT-J03WN1 F4SPM_DUMP=<dir> python -I candidates/bench.py candidates/final_f4spm/dump_members.py \
        --dfl2 --folds <folds json> --save-probs $MTAH_WORK/probs/r5_f4spm_live_dfl
    F4SPM_DUMP=<dir> python -I candidates/bench.py candidates/final_f4spm/dump_members.py --save-probs $MTAH_WORK/probs/r5_f4spm_live_met
    F4SPM_PREFIX=skc_<game> F4SPM_DUMP=<dir> python -I candidates/nn_pointer/skc_both.py \
        candidates/final_f4spm/dump_members.py r5_f4spm_live_skc --games <game>
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("f4spm_dump_base", os.path.join(HERE, "pipeline.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
OUT = os.environ["F4SPM_DUMP"]
PREFIX = os.environ.get("F4SPM_PREFIX") or os.environ.get("BENCH_DFL_ONLY") or "dev"
_N = [0]


def predict(*a):
    codes = base.predict(*a)
    os.makedirs(OUT, exist_ok=True)
    for i, m in enumerate(base._VOTE.members):
        np.save(os.path.join(OUT, "{}_{}_m{}.npy".format(PREFIX, _N[0], i)), np.asarray(m.mod.predict_proba(*a), np.float32))
        np.save(os.path.join(OUT, "{}_{}_m{}_codes.npy".format(PREFIX, _N[0], i)), np.asarray(m.mod.predict(*a), np.int8))
    _N[0] += 1
    return codes


predict_proba = base.predict_proba
