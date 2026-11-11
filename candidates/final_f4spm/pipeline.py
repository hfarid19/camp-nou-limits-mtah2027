"""final_f4spm ("F4sP-MF2", submission candidate v9; docs/prereg/round5_prereg.txt addenda of 11 Nov 12:31 and 12:46):
candidates/final_f4sp/pipeline.py (F4sP, submission v8) with its MS-TCN slot swapped for the Metrica fine-tuned member;
everything else, including the decoder config, is F4sP's. MBR decode (candidates/mbr_decode) of four members, then
prekick_dead:

    TCN slot      adapt/member_tcn_tta4_a50.py           o_ls05 + test-time averaging (4 jittered copies) + all_a50 renorm
    MS-TCN slot   nn_pff/pipeline_c192ls_skcft_metft.py  m5c_C192ls_skcft_metft: v8's MS-TCN (m5c_C192ls_skcft) fine-tuned
                                                         with Metrica Sample Games 1-3 added (MET-FT recipe, lr 5e-5)
    radar slot    ball_radar_pff/pipeline.py             radar_pff
    pointer slot  nn_pointer/pipeline_ballc_p20m.py      r5_ptr_ballc_p20M (19.3M-parameter ball-centred pointer)
Decoder config = the full-DFL pick of the g1 grid for this member set (finals.py; the same as F4sP's): src hmm,
alpha 0.1, rho 0.2, margin 0.0; fixed sampling seed (decoder.SEED). Member probabilities are rounded to float32 first
(pipeline_mbr.MBRVote), so the live decode equals the offline one on saved probabilities. Numpy + LightGBM (+ PyTorch
on the CPU for the pointer's forward pass); predict / predict_proba follow the bench interface.
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
           ("nn_pff", "pipeline_c192ls_skcft_metft.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc_p20m.py")]
CFG = dict(src="hmm", alpha=0.1, rho=0.2, margin=0.0)
_VOTE = pipeline_mbr.MBRVote(MEMBERS, CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
