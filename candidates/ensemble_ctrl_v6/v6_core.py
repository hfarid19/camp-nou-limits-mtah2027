"""ensemble_ctrl_v6: v5c's vote machinery with the member list and the rule as parameters.

Vote(members, smooth_s, control_s) builds what ensemble_ctrl_v5c/pipeline.py does with its constants:
hard majority vote of the members' own predict (tie-break = member order, ensemble_core.hard_vote),
learned_dfl_evidence's merge_short_v2 at smooth_s (off if 0), control.fold at control_s, then
prekick_dead (every graded frame before a half's kick-off is D: Smart Tagging "Kick Off (Start)" + 88
frames; no-op without tags). predict / predict_proba follow the bench interface.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (CAND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import control              # noqa: E402
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402

KICK_LAG = 88


def prekick_dead(codes, tags, blocks):
    out = codes.copy()
    for _, first, last in blocks:
        for code, s, e, _team, kind in tags:
            if code == "set pieces" and "kick off (start)" in str(kind).lower() and first <= s <= last:
                kick = min(s + KICK_LAG, last + 1)
                out[first:kick] = 2
                break
    return out


class Vote(object):
    def __init__(self, member_specs, smooth_s, control_s):
        self.members = [members.Member(s, f) for s, f in member_specs]
        self.lde = members.load("learned_dfl_evidence")                 # merge_short_v2 only
        self.smooth = dict(min_poss_s=smooth_s, min_dead_s=smooth_s) if smooth_s > 0 else None
        self.control_s = control_s
        self.cache = {}

    def run(self, args):
        key = members.input_key(args)
        if key in self.cache:
            return self.cache[key]
        self.cache.clear()
        blocks, n, fps = args[4], len(args[2]), int(args[8])
        codes_list = [m.run(args)[0] for m in self.members]
        if len(codes_list) == 1 and self.smooth is None:
            codes = codes_list[0].copy()
        else:
            codes = ec.hard_vote(codes_list, blocks, self.lde, fps, None, self.smooth)
        runs = [(int(first), int(last) + 1) for _, first, last in blocks]
        codes = control.fold(np.asarray(codes, np.int8), self.control_s, runs, fps)
        codes = prekick_dead(codes, args[5], blocks)
        mask = ec.block_mask(blocks, n)
        P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
        self.cache[key] = (np.asarray(codes, np.int8), P)
        return self.cache[key]
