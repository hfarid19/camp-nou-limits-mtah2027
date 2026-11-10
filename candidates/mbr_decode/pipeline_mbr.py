"""mbr_decode on the lsw members: expected-spellF1 (minimum-Bayes-risk) window selection instead of the
fixed v5c rule alone. See README.md.

Members (tie-break order as pipeline_lsw.py):
    nn_pff_tcn/pipeline_ls.py   label-smoothed TCN       (o_ls05)
    nn_pff/pipeline_w.py        wide MS-TCN              (m5c_W)
    ball_radar_pff/pipeline.py  radar + stacked LightGBM (radar_pff)
Decode (decoder.decode, config CFG chosen on DFL leave-one-game-out, nested):
    candidates = the hard vote under the v5c rule (the base = pipeline_lsw.py's codes before prekick_dead),
                 the vote under 11 other (merge_short, fold) rules, each member alone (raw, v5c rule);
    samples    = 32 FFBS paths of a tempered 3-label HMM on the members' mean probabilities, each read
                 at fold 0 / 0.5 / 1 s;
    per window (stretch between frames where every candidate says D) the candidate maximising
                 2 * expected matched spells - F * spells (+ margin for leaving the base), F iterated;
then prekick_dead, as v6. predict / predict_proba follow the bench interface (proba = vote shares).
Deterministic: numpy only, fixed seed (decoder.SEED); member probabilities are rounded to float32 first
so the live decode equals the offline one on the benches' saved probabilities.
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402
import v6_core              # noqa: E402
import decoder              # noqa: E402

MEMBERS = [("nn_pff_tcn", "pipeline_ls.py"),
           ("nn_pff", "pipeline_w.py"),
           ("ball_radar_pff", "pipeline.py")]
CFG = dict(src="hmm", alpha=1.0, rho=0.2, margin=0.5)     # full-DFL pick of grid g1 (README.md)


class MBRVote(object):
    def __init__(self, member_specs, cfg):
        self.members = [members.Member(s, f) for s, f in member_specs]
        self.cfg = dict(cfg)
        self.cache = {}

    def run(self, args):
        key = members.input_key(args)
        if key in self.cache:
            return self.cache[key]
        self.cache.clear()
        blocks, n = args[4], len(args[2])
        codes_list, probs = [], []
        for m in self.members:
            codes_list.append(np.asarray(m.mod.predict(*args), np.int8))
            probs.append(np.asarray(m.mod.predict_proba(*args), np.float32).astype(np.float64))
        codes, _ = decoder.decode(probs, codes_list, blocks, self.cfg)
        codes = v6_core.prekick_dead(np.asarray(codes, np.int8), args[5], blocks)
        mask = ec.block_mask(blocks, n)
        P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
        self.cache[key] = (np.asarray(codes, np.int8), P)
        return self.cache[key]


_VOTE = MBRVote(MEMBERS, CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
