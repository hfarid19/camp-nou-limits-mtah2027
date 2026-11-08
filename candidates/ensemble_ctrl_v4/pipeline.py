"""ensemble_ctrl_v4: ensemble_ctrl_v2's vote with members trained on the raw DFL flag, and a
1.25 s control fold, chosen for the ground truth's fine possession granularity.

Hard majority vote of MEMBERS (each member's own predict; tie-break to the earlier member, as
ensemble_vote), learned_dfl_evidence's merge_short_v2 at 1.0 s, then control.fold(codes,
CONTROL_S = 1.25 s), then prekick_dead (before a half's kick-off, play is dead).

Why: EVALUATION.md publishes the full-match spellF1 of a perfect submission shifted 3 / 8 / 25
frames late (0.982 / 0.893 / 0.654). That fingerprint depends only on the truth's spell lengths;
it lies between the raw DFL flag and a 1 s fold, far from the 2 s fold the v2 members were trained
on (0.999 / 0.989 / 0.797). The member variants and the fold were chosen by nested
leave-one-game-out selection on fine4 (candidates/score_sel2.py). See README.md.

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

MEMBERS = [("temporal_tcn_v2", "pipeline_raw.py"),             # tie-break order
           ("handover_mstcn_v2", "pipeline_raw.py"),
           ("ball_radar", "pipeline.py")]
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
