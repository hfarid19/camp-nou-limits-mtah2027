"""final_f4sps ("F4sP + S2", submission CANDIDATE v10; docs/prereg/round5_prereg.txt addendum of 13 Nov 2026 09:02 CST,
candidate "S2a"): candidates/final_f4sp/pipeline.py (F4sP, submission v8) with a FIFTH member in the same MBR decode: the
Metrica-trained stacker S2 (candidates/stacker_met), a LightGBM model whose 84 inputs are the four members' probabilities
at 7 time offsets. Members, weights, decoder config and post-processing are v8's:

    TCN slot      adapt/member_tcn_tta4_a50.py         o_ls05 + test-time averaging (4 jittered copies) + all_a50 renorm
    MS-TCN slot   nn_pff/pipeline_c192ls_skcft.py      m5c_C192ls_skcft (v8's; NOT v9's Metrica fine-tuned MS-TCN)
    radar slot    ball_radar_pff/pipeline.py           radar_pff
    pointer slot  nn_pointer/pipeline_ballc_p20m.py    r5_ptr_ballc_p20M (19.3M-parameter ball-centred pointer)
    stacker       stacker_met/stacker.py               S2 on the four members' float32 probabilities (stacker_s2.txt)

Decoder config = v8's: src hmm, alpha 0.1, rho 0.2, margin 0.0; fixed sampling seed (decoder.SEED); the stacker's
probabilities and codes enter decoder.decode as the fifth member, after the four, exactly as the experiment's scoring
(arch/r5_arch/stack_aug/code/score_S.py, bench_job). Member probabilities are rounded to float32 before the stacker and the
decode; the stacker evaluates every frame inside the graded blocks and its probabilities are rounded to float32 too, so
the live decode equals the offline one on saved probabilities. Then prekick_dead, as v8. predict_proba = vote shares of
the five members' codes (not used by make_submission.py).
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), os.path.join(CAND, "mbr_decode"), os.path.join(CAND, "stacker_met"),
           HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402
import v6_core              # noqa: E402
import decoder              # noqa: E402
import stacker              # noqa: E402

MEMBERS = [("adapt", "member_tcn_tta4_a50.py"),
           ("nn_pff", "pipeline_c192ls_skcft.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc_p20m.py")]
CFG = dict(src="hmm", alpha=0.1, rho=0.2, margin=0.0)


class StackVote(object):
    """pipeline_mbr.MBRVote with the stacker as a fifth decode member. self.last keeps the latest unit's member outputs
    (four members, then the stacker) for dump_members.py."""

    def __init__(self, member_specs, cfg):
        self.members = [members.Member(s, f) for s, f in member_specs]
        self.cfg = dict(cfg)
        self.cache = {}
        self.last = None
        self.booster = None

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
        if self.booster is None:
            self.booster = stacker.load()
        sp, sc = stacker.predict(probs, blocks, self.booster)
        codes_list.append(sc)
        probs.append(np.asarray(sp, np.float32).astype(np.float64))
        self.last = (list(codes_list), list(probs))
        codes, _ = decoder.decode(probs, codes_list, blocks, self.cfg)
        codes = v6_core.prekick_dead(np.asarray(codes, np.int8), args[5], blocks)
        mask = ec.block_mask(blocks, n)
        P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
        self.cache[key] = (np.asarray(codes, np.int8), P)
        return self.cache[key]


_VOTE = StackVote(MEMBERS, CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
