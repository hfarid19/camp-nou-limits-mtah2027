"""final_f4b (round 5 final recipe, F4b; docs/prereg/round5_prereg.txt addenda): MBR decode (candidates/mbr_decode) of
four members, then prekick_dead, as candidates/adapt/pipeline_mbr_tta4_a50.py with the round-5 member slots:

    TCN slot      adapt/member_tcn_tta4_a50.py   o_ls05 + test-time averaging (4 jittered copies) + all_a50 renorm
    MS-TCN slot   nn_pff/pipeline_c192ls.py      m5c_C192ls (best DFL + SkillCorner gain alone vs m5c_W)
    radar slot    ball_radar_pff/pipeline.py     radar_pff
    pointer slot  nn_pointer/pipeline_ballc.py   r5_ptr_ballc (passed the SkillCorner-alone gate vs o_ls05)
Decoder config = the full-DFL pick of the g1 grid for this member set (candidates/nn_pointer/finals.py):
src hmm, alpha 1.0, rho 0.2, margin 0.0; fixed sampling seed (decoder.SEED). Member probabilities are rounded to
float32 first (pipeline_mbr.MBRVote), so the live decode equals the offline one on saved probabilities.
Numpy + LightGBM only; predict / predict_proba follow the bench interface.
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
           ("nn_pff", "pipeline_c192ls.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc.py")]
CFG = dict(src="hmm", alpha=1.0, rho=0.2, margin=0.0)
_VOTE = pipeline_mbr.MBRVote(MEMBERS, CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
