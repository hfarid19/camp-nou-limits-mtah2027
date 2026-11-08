"""The committed possession_pipeline.py, unchanged, behind the bench's predict() interface."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import possession_pipeline as pp  # noqa: E402


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P = dict(pp.PARAMS)
    E, aux = pp.build_evidence(players, ball, slot_team, sign_half1, blocks, tags, names, pitch,
                               P, lambda *a, **k: None)
    out = np.full(len(ball), 3, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        path = pp.viterbi(E[seg], P)
        path = pp.merge_short(path, P)
        path = pp.snap_boundaries(path, {k: v[seg] for k, v in aux.items()}, P)
        out[seg] = path
    return out
