"""selftrain_ctrl: ensemble_ctrl + a single decision tree (CART, depth 8) self-trained on the scored game's own pseudo-labels.
See selftrain_core.py. predict(...) -> codes, predict_proba(...) -> (N, 3); bench.py interface."""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selftrain_core  # noqa: E402

_ST = selftrain_core.SelfTrain(model="tree", params=dict(max_depth=8, min_samples_leaf=50), min_dist=25, stride=5, w=1.5)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _ST.predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _ST.predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
