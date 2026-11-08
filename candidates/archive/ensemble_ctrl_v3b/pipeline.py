"""ensemble_ctrl_v3b: ensemble_ctrl_v2's vote with a learned spell-level control rule instead of the 2 s fold.

Hard majority vote of SLUGS and learned_dfl_evidence's merge_short_v2 at 1.0 s, exactly as
ensemble_ctrl_v2. Then, instead of control.fold(codes, 2.0 s), every contested spell (a team spell
directly following the other team's spell) is kept or handed back to the preceding team by a
LightGBM classifier P(keep | spell features) (spellctl.py), with control.fold's pass structure.
Model: model.txt; threshold, pass mode and training variant: config.json (both trained on the 7 DFL
games by train.py). Under bench --dfl2 both are swapped for the leave-one-game-out fold files
(LightGBM by the bench's Booster swap, config.json through dfl_folds.resolve).

numpy / scipy / lightgbm only. predict(...) -> codes, predict_proba(...) -> (N, 3) vote shares.
"""

import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
V2 = os.path.join(CAND, "ensemble_ctrl_v2")
for _p in (CAND, V2, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lightgbm as lgb      # noqa: E402
import dfl_folds            # noqa: E402
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402
import spellctl             # noqa: E402

SLUGS = ["temporal_tcn_v2", "handover_mstcn_v2", "ball_radar"]     # tie-break order
SMOOTH = dict(min_poss_s=1.0, min_dead_s=1.0)

_MEMBERS = {s: members.Member(s) for s in SLUGS}
_LDE = members.load("learned_dfl_evidence")                       # merge_short_v2 only
_CACHE = {}
_MODEL = {}


def _model():
    if not _MODEL:
        _MODEL["booster"] = lgb.Booster(model_file=os.path.join(HERE, "model.txt"))
        with open(dfl_folds.resolve(os.path.join(HERE, "config.json"))) as h:
            _MODEL["cfg"] = json.load(h)
    return _MODEL["booster"], _MODEL["cfg"]


def _run(args):
    key = members.input_key(args)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    players, slot_team, ball, sign, blocks, pitch, fps = (args[0], args[1], args[2], args[3], args[4],
                                                          args[7], int(args[8]))
    n = len(ball)
    outs = [_MEMBERS[s].run(args) for s in SLUGS]
    codes_list = [o[0] for o in outs]
    codes = np.asarray(ec.hard_vote(codes_list, blocks, _LDE, fps, None, SMOOTH), np.int8)
    booster, cfg = _model()
    F = spellctl.Frames(players, slot_team, ball, sign, blocks, codes_list,
                        [np.asarray(o[1], np.float32) for o in outs], tuple(pitch), fps)
    codes = spellctl.apply(F, codes, blocks, booster.predict, float(cfg["thr"]),
                           iterate=(cfg["mode"] == "iter"))
    mask = ec.block_mask(blocks, n)
    P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
    _CACHE[key] = (np.asarray(codes, np.int8), P)
    return _CACHE[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
