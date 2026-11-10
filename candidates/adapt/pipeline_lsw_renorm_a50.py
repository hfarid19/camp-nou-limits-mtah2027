"""lsw (pipeline_lsw.py) with its two NN members re-normalised per game (renorm.py variant all_a50: the
features' training mean / sd moved halfway to the game's own mirror-pooled statistics over
its graded blocks); radar_pff unchanged; v6's rule (merge_short 1 s, control.fold 1.25 s, prekick_dead).
predict / predict_proba follow the bench interface.
"""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v6_core  # noqa: E402

MEMBERS = [("adapt", "member_tcn_a50.py"),
           ("adapt", "member_mstcn_a50.py"),
           ("ball_radar_pff", "pipeline.py")]
_VOTE = v6_core.Vote(MEMBERS, 1.0, 1.25)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
