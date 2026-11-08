"""ensemble_ctrl_v2: ensemble_ctrl with the sequence members retrained on the realistic degradation.

Hard majority vote of SLUGS (each member's own predict; tie-break to the earlier member, as
ensemble_vote), learned_dfl_evidence's merge_short_v2 at 1.0 s (ensemble_vote's "m10"), then the
challenge's control rule on the output: control.fold(codes, CONTROL_S = 2.0 s), exactly as
ensemble_ctrl. The rule is fixed in this file (no DFL-fitted config file), so only the members'
own models need fold models under bench --dfl / --dfl2. See README.md.

predict(...) -> codes, predict_proba(...) -> (N, 3) vote shares; bench.py interface.
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (CAND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import control              # noqa: E402
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402

SLUGS = ["temporal_tcn_v2", "handover_mstcn_v2", "ball_radar"]     # tie-break order
SMOOTH = dict(min_poss_s=1.0, min_dead_s=1.0)
CONTROL_S = 2.0

_MEMBERS = {s: members.Member(s) for s in SLUGS}
_LDE = members.load("learned_dfl_evidence")                       # merge_short_v2 only
_CACHE = {}


def _run(args):
    key = members.input_key(args)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    blocks, n, fps = args[4], len(args[2]), int(args[8])
    codes_list = [_MEMBERS[s].run(args)[0] for s in SLUGS]
    codes = ec.hard_vote(codes_list, blocks, _LDE, fps, None, SMOOTH)
    runs = [(int(first), int(last) + 1) for _, first, last in blocks]
    codes = control.fold(np.asarray(codes, np.int8), CONTROL_S, runs, fps)
    mask = ec.block_mask(blocks, n)
    P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
    _CACHE[key] = (np.asarray(codes, np.int8), P)
    return _CACHE[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
