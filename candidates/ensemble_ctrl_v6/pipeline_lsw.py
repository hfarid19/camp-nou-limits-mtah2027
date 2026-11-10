"""ensemble_ctrl_v6, candidate "lsw" (post-hoc, NOT frozen; for the user to decide): the frozen v6
(pipeline.py: v5c+PFF members + v5c's rule) with two members replaced by variants found in the
optimisation and width x context experiments (candidates/nn_pff/README.md):
    nn_pff_tcn/pipeline_ls.py   the t5c TCN recipe with label smoothing 0.05 (weights tcn_ls.npz)
    nn_pff/pipeline_w.py        the m5c MS-TCN recipe at width C 128 / Cr 64 (weights mstcn_w.npz)
    ball_radar_pff/pipeline.py  unchanged
Rule: v5c's (merge_short_v2 1 s, control.fold 1.25 s, prekick_dead), the frozen v6's rule.
Chosen after seeing the results (no holdout left to confirm it): see README.md.
"""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v6_core  # noqa: E402

MEMBERS = [("nn_pff_tcn", "pipeline_ls.py"),
           ("nn_pff", "pipeline_w.py"),
           ("ball_radar_pff", "pipeline.py")]
SMOOTH_S = 1.0
CONTROL_S = 1.25

_VOTE = v6_core.Vote(MEMBERS, SMOOTH_S, CONTROL_S)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
