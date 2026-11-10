"""ensemble_ctrl_v5a: ensemble_ctrl_v4 plus a short-possession recovery step from the members' per-frame
probabilities. Built to test whether the short spells the truth keeps (35% under 3 s) can be recovered;
on the bench they cannot (nested fine4 -0.001 +- 0.002, see README.md). NOT recommended over v4.

Steps, in order (1-4 identical to ensemble_ctrl_v4):
1. hard majority vote of temporal_tcn_v2 (raw) / handover_mstcn_v2 (raw) / ball_radar, tie-break in that order;
2. learned_dfl_evidence merge_short_v2 at 1.0 s;
3. control.fold at 1.25 s;
4. prekick_dead (match only);
5. insert_short: inside every predicted alive spell of team c, each run of >= LMIN frames where the mean of the
   two TCN members' P(opponent) exceeds THR becomes an opponent spell. Chosen by nested leave-one-game-out
   selection on fine4 over source x THR x LMIN x smoothing (candidates/_scratch/v5a_soft.py).

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
SOFT_MEMBERS = (0, 1)   # members whose P(H, A, D) are pooled (arithmetic mean) for insert_short
THR = 0.8               # pooled P(opponent) above this ...
LMIN = 15               # ... for at least this many frames (0.6 s) makes an opponent spell

_MEMBERS = [members.Member(s, f) for s, f in MEMBERS]
_LDE = members.load("learned_dfl_evidence")                       # merge_short_v2 only
_CACHE = {}


def prekick_dead(codes, tags, blocks):
    """(ensemble_ctrl_v4) Every graded frame of a half before its kick-off is D; the kick is KICK_LAG frames
    after the start of the half's "Kick Off (Start)" tag. No-op without tags (DFL, Metrica benches)."""
    out = codes.copy()
    for _, first, last in blocks:
        for code, s, e, _team, kind in tags:
            if code == "set pieces" and "kick off (start)" in str(kind).lower() and first <= s <= last:
                kick = min(s + KICK_LAG, last + 1)
                out[first:kick] = 2
                break
    return out


def insert_short(codes, P, runs, thr=THR, lmin=LMIN):
    """Inside each predicted H/A spell, runs of >= lmin frames with P[:, opponent] > thr become the opponent."""
    import possession_metrics as pm
    out = codes.copy()
    for s, e, c0 in pm.segments(codes, runs):
        if c0 not in (0, 1):
            continue
        opp = 1 - c0
        m = P[s:e, opp] > thr
        if not m.any():
            continue
        d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= lmin:
                out[s + a:s + b] = opp
    return out


def _run(args):
    key = members.input_key(args)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    blocks, n, fps = args[4], len(args[2]), int(args[8])
    outs = [m.run(args) for m in _MEMBERS]
    codes_list = [o[0] for o in outs]
    codes = ec.hard_vote(codes_list, blocks, _LDE, fps, None, SMOOTH)
    runs = [(int(first), int(last) + 1) for _, first, last in blocks]
    codes = control.fold(np.asarray(codes, np.int8), CONTROL_S, runs, fps)
    P = np.mean([np.asarray(outs[k][1], np.float64) for k in SOFT_MEMBERS], axis=0)
    codes = insert_short(codes, P, runs)
    codes = prekick_dead(codes, args[5], blocks)
    mask = ec.block_mask(blocks, n)
    Pv = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
    _CACHE[key] = (np.asarray(codes, np.int8), Pv)
    return _CACHE[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
