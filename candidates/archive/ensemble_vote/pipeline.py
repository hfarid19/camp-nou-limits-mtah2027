"""Ensemble of the possession candidates, hard majority vote, the shipped variant (see README.md and vote_pipeline.py).

predict(...) -> codes, predict_proba(...) -> (N, 3) P(H), P(A), P(D); bench.py interface."""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vote_pipeline  # noqa: E402

_VOTE = vote_pipeline.Vote(os.path.join(HERE, "config_hard.json"))


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
