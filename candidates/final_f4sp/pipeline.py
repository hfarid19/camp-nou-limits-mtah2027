"""final_f4sp (round 5, "F4sP"; docs/prereg/round5_prereg.txt addenda 11:20 / 11:50 and 13:30 / 14:55 of 10 Nov):
MBR decode (candidates/mbr_decode) of four members, then prekick_dead, exactly as candidates/final_f4b/pipeline.py
with two member slots swapped and the decoder config re-picked for the new member set:

    TCN slot      adapt/member_tcn_tta4_a50.py         o_ls05 + test-time averaging (4 jittered copies) + all_a50 renorm
    MS-TCN slot   nn_pff/pipeline_c192ls_skcft.py      m5c_C192ls_skcft: m5c_C192ls fine-tuned with the 20 SkillCorner
                                                       games added (passed the +SKC member gate)
    radar slot    ball_radar_pff/pipeline.py           radar_pff
    pointer slot  nn_pointer/pipeline_ballc_p20m.py    r5_ptr_ballc_p20M: the ball-centred pointer widened to 19.3M
                                                       parameters (passed the pointer size-ladder member gate)
Decoder config = the full-DFL pick of the g1 grid for this member set (candidates/nn_pointer/finals.py; the same
pick in all 7 nested folds): src hmm, alpha 0.1, rho 0.2, margin 0.0; fixed sampling seed (decoder.SEED). Member
probabilities are rounded to float32 first (pipeline_mbr.MBRVote), so the live decode equals the offline one on
saved probabilities. Numpy + LightGBM only; predict / predict_proba follow the bench interface.
"""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), os.path.join(CAND, "mbr_decode"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pipeline_mbr  # noqa: E402

MEMBERS = [("adapt", "member_tcn_tta4_a50.py"),
           ("nn_pff", "pipeline_c192ls_skcft.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc_p20m.py")]
CFG = dict(src="hmm", alpha=0.1, rho=0.2, margin=0.0)
_VOTE = pipeline_mbr.MBRVote(MEMBERS, CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
