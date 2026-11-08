"""ensemble_ctrl_v5c: ensemble_ctrl_v4's vote with all three members retrained on corrected data.

Hard majority vote of MEMBERS (each member's own predict; tie-break to the earlier member, as
ensemble_vote), learned_dfl_evidence's merge_short_v2 at 1.0 s, then control.fold(codes,
CONTROL_S = 1.25 s), then prekick_dead (before a half's kick-off, play is dead) -- v4's rule,
unchanged (a nested leave-one-game-out re-selection of smoothing x fold on fine4 picks it again in
6 of 7 folds).

What changed: the members. temporal_tcn_v3, handover_mstcn_v3 and ball_radar_v3 are
temporal_tcn_v2 (raw), handover_mstcn_v2 (raw) and ball_radar with the same code, architecture,
targets and training recipes, retrained after fixing the attack direction of one training game.
The training sets took `sign` (does the home team attack +x in the first half?) from
learned_dfl_evidence.make_dataset.attack_sign, which compares the home team's mean x over the WHOLE
first half with the halfway line. For DFL-MAT-J03WR9 that is 54.9 m > 52.5 m -> -1, but at the
first-half kick-off the home team stands at 45.7 m: it attacks +x (+1, the bench's rule,
tune_on_sample_data.attack_sign). So one of the 7 training games had every direction-dependent
feature mirrored. See README.md.

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

MEMBERS = [("temporal_tcn_v3", "pipeline.py"),                # tie-break order
           ("handover_mstcn_v3", "pipeline.py"),
           ("ball_radar_v3", "pipeline.py")]
SMOOTH = dict(min_poss_s=1.0, min_dead_s=1.0)
CONTROL_S = 1.25
KICK_LAG = 88    # frames from a "Kick Off (Start)" tag's start to the kick, from the first half

_MEMBERS = [members.Member(s, f) for s, f in MEMBERS]
_LDE = members.load("learned_dfl_evidence")                       # merge_short_v2 only
_CACHE = {}


def prekick_dead(codes, tags, blocks):
    """Every graded frame of a half before its kick-off is D (FAQ: the second half is graded from
    110,808 but kicks off at 113,742). The kick is placed KICK_LAG frames after the start of the
    half's "Kick Off (Start)" Smart Tagging instance: in the first half, which is graded from the
    kick itself, the tag starts 88 frames before it. No-op without tags (DFL, Metrica benches)."""
    out = codes.copy()
    for _, first, last in blocks:
        for code, s, e, _team, kind in tags:
            if code == "set pieces" and "kick off (start)" in str(kind).lower() and first <= s <= last:
                kick = min(s + KICK_LAG, last + 1)
                out[first:kick] = 2
                break
    return out


def _run(args):
    key = members.input_key(args)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    blocks, n, fps = args[4], len(args[2]), int(args[8])
    codes_list = [m.run(args)[0] for m in _MEMBERS]
    codes = ec.hard_vote(codes_list, blocks, _LDE, fps, None, SMOOTH)
    runs = [(int(first), int(last) + 1) for _, first, last in blocks]
    codes = control.fold(np.asarray(codes, np.int8), CONTROL_S, runs, fps)
    codes = prekick_dead(codes, args[5], blocks)
    mask = ec.block_mask(blocks, n)
    P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
    _CACHE[key] = (np.asarray(codes, np.int8), P)
    return _CACHE[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
