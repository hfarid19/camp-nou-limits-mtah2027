"""ensemble_ctrl_v5b: ensemble_ctrl_v4 plus a dead-ball clean-up chosen on the match-calibrated
bench (degrade_v4, see bench_v4.py and README.md).

Steps (1-4 are ensemble_ctrl_v4 unchanged): hard vote of temporal_tcn_v2 (raw flag variant),
handover_mstcn_v2 (raw) and ball_radar; learned_dfl_evidence's merge_short_v2 at 1.0 s;
control.fold at CONTROL_S; prekick_dead. Step 5, dead_cleanup, is the only addition:

  MIN_D_S   a D run shorter than MIN_D_S with the same team on both sides is handed back to that
            team (a stoppage that short with no change of possession is a flicker, not a restart);
            with different teams on the two sides it is given to the following team (after a
            restart there is a clear possessor, and the fold has already judged that spell);
  JOIN_D_S  a team spell shorter than JOIN_D_S between two D runs becomes D (the ball shown for a
            moment during one stoppage is not a possession).
Both default to 0 (no-op) unless README.md's nested selection set them.

predict(...) -> codes, predict_proba(...) -> (N, 3) vote shares; bench.py interface.
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
for _p in (ROOT, os.path.join(ROOT, "evaluation"), CAND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import possession_metrics as pm  # noqa: E402
import control              # noqa: E402
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402

MEMBERS = [("temporal_tcn_v2", "pipeline_raw.py"),             # tie-break order
           ("handover_mstcn_v2", "pipeline_raw.py"),
           ("ball_radar", "pipeline.py")]
SMOOTH = dict(min_poss_s=1.0, min_dead_s=1.0)
CONTROL_S = 1.25
KICK_LAG = 88    # frames from a "Kick Off (Start)" tag's start to the kick, from the first half
MIN_D_S = 0.0
JOIN_D_S = 2.0

_MEMBERS = [members.Member(s, f) for s, f in MEMBERS]
_LDE = members.load("learned_dfl_evidence")                       # merge_short_v2 only
_CACHE = {}


def prekick_dead(codes, tags, blocks):
    """Every graded frame of a half before its kick-off is D (ensemble_ctrl_v4.prekick_dead)."""
    out = codes.copy()
    for _, first, last in blocks:
        for code, s, e, _team, kind in tags:
            if code == "set pieces" and "kick off (start)" in str(kind).lower() and first <= s <= last:
                kick = min(s + KICK_LAG, last + 1)
                out[first:kick] = 2
                break
    return out


def dead_cleanup(codes, runs, min_d_s, join_d_s, fps):
    out = codes.copy()
    if min_d_s:
        for a, b, c in pm.segments(out, runs):
            if c != pm.D or b - a >= min_d_s * fps:
                continue
            prev = out[a - 1] if a > 0 and out[a - 1] in (pm.H, pm.A) else None
            nxt = out[b] if b < len(out) and out[b] in (pm.H, pm.A) else None
            if prev is None and nxt is None:
                continue
            out[a:b] = nxt if nxt is not None else prev
    if join_d_s:
        segs = pm.segments(out, runs)
        for i in range(1, len(segs) - 1):
            a, b, c = segs[i]
            if c in (pm.H, pm.A) and b - a < join_d_s * fps and segs[i - 1][2] == pm.D \
                    and segs[i + 1][2] == pm.D and segs[i - 1][1] == a and segs[i + 1][0] == b:
                out[a:b] = pm.D
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
    codes = dead_cleanup(codes, runs, MIN_D_S, JOIN_D_S, fps)
    mask = ec.block_mask(blocks, n)
    P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
    _CACHE[key] = (np.asarray(codes, np.int8), P)
    return _CACHE[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
