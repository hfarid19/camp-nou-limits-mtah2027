"""ensemble_ctrl_v6, candidate "v5c+PFF": ensemble_ctrl_v5c's three members with their exact recipes,
retrained on the 7 DFL games + 48 PFF World Cup 2022 games (touch labels), voted with v5c's machinery.

Members (tie-break order = DFL leave-one-game-out fine4, best first):
    nn_pff_tcn/pipeline_v5cpff.py   temporal_tcn_v3's TCN + LDE decoder   (weights tcn_v5cpff.npz)
    nn_pff/pipeline_v5cpff.py       handover_mstcn_v3's MS-TCN + ASRF decoder (mstcn_v5cpff.npz)
    ball_radar_pff/pipeline.py      ball_radar_v3's radar + stacked LightGBM (radar.npz, model_br*.txt)
Rule (nested choice, see README.md): no merge_short smoothing, control.fold 0.75 s, prekick_dead.

predict(...) -> codes, predict_proba(...) -> (N, 3) vote shares; bench.py interface.
"""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v6_core  # noqa: E402

MEMBERS = [("nn_pff_tcn", "pipeline_v5cpff.py"),
           ("nn_pff", "pipeline_v5cpff.py"),
           ("ball_radar_pff", "pipeline.py")]
SMOOTH_S = 0.0
CONTROL_S = 0.75

_VOTE = v6_core.Vote(MEMBERS, SMOOTH_S, CONTROL_S)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
