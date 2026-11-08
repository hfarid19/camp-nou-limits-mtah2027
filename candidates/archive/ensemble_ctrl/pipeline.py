"""ensemble_ctrl: ensemble_vote with the challenge's control rule applied to its own output.

CHALLENGE.md counts a change of possession only when the opponent gains CONTROLLED possession
(a touch, a deflection or a won aerial duel without control does not count; proxy: "a player keeps
the ball for a few seconds"). The models were trained on DFL labels that flip on every touch, so
their output still contains short opponent spells the competition would not count. This candidate
hands every predicted possession spell shorter than CONTROL_S seconds back to the team before it
(control.fold); spells after a stoppage are kept.

CONTROL_S = 2.0 was chosen on DFL leave-one-game-out (bench --dfl2) as the value with the best mean
spellF1 over controlled-possession labels with T = 1, 2, 3 s (0.728 vs 0.696 unfolded), and the
smallest worst-case loss over those T (0.048). It costs 0.034 against raw every-touch labels,
which the definition rules out. No competition label was used.
"""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.insert(0, CAND)
import control                  # noqa: E402

CONTROL_S = 2.0


def _load_ensemble():
    path = os.path.join(CAND, "ensemble_vote", "pipeline.py")
    spec = importlib.util.spec_from_file_location("ensemble_ctrl_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    return module


_BASE = _load_ensemble()


def _runs(n, blocks):
    return [(int(first), int(last) + 1) for _, first, last in blocks]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    codes = np.asarray(_BASE.predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps),
                       np.int8)
    return control.fold(codes, CONTROL_S, _runs(len(codes), blocks), fps)


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _BASE.predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
