"""R5-1's MBR decode (mbr_decode/pipeline_mbr.py, its fixed CFG) on the lsw members with the TCN member replaced by
its test-time-averaged version (member_tcn_tta4.py: 4 jittered input copies + the original, probabilities
averaged). MS-TCN m5c_W and radar_pff unchanged. Then prekick_dead.
predict / predict_proba follow the bench interface.
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

MEMBERS = [("adapt", "member_tcn_tta4.py"),
           ("nn_pff", "pipeline_w.py"),
           ("ball_radar_pff", "pipeline.py")]
_VOTE = pipeline_mbr.MBRVote(MEMBERS, pipeline_mbr.CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
