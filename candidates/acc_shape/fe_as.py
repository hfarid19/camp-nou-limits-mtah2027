"""Drop-in replacement for the `fe` module of nn_pff_tcn/pipeline.py and nn_pff/pipeline.py: the repo features
(candidates/nn_pff/features.py, unchanged) with the 29 ACC + SHAPE columns of nf_features.py appended (ACC first, then
SHAPE), float32. swap_index / frame_sign are the repo's.

The 29 extra columns of one input (players, slot_team, ball, pitch, fps; they do not depend on the attack sign) are kept
in memory for the last few inputs, so the MS-TCN member reuses the pass the TCN member already made on the unjittered
input. A cache hit returns the array that was computed, so the output does not depend on the cache. Nothing is written
to disk. _common.fe() loads it once per process, so both members see the same instance.
"""
import hashlib
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("mtah_acc_shape_nf_features", os.path.join(HERE, "nf_features.py"))
nf = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(nf)
swap_index = nf.fe.swap_index
frame_sign = nf.fe.frame_sign
N_EXTRA = 29
KEEP = 6                 # inputs kept (one TTA member pass = the original + 4 jittered copies)
_CACHE = {}              # key -> (E, names, kinds), insertion-ordered


def _key(players, slot_team, ball, pitch, fps):
    h = hashlib.sha1()
    for a in (players, slot_team, ball):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    h.update(repr((tuple(pitch), fps)).encode())
    return h.hexdigest()


def extra(players, slot_team, ball, pitch, fps):
    k = _key(players, slot_team, ball, pitch, fps)
    if k not in _CACHE:
        while len(_CACHE) >= KEEP:
            del _CACHE[next(iter(_CACHE))]
        E, en, ek = nf.extra(players, slot_team, ball, pitch, fps, nf.GROUPS)
        assert E.shape[1] == N_EXTRA
        _CACHE[k] = (E, en, ek)
    return _CACHE[k]


def compute(players, slot_team, ball, sign, pitch, fps):
    F, names, kinds = nf.fe.compute(players, slot_team, ball, sign, pitch, fps)
    E, en, ek = extra(players, slot_team, ball, pitch, fps)
    return np.concatenate([F, E], 1).astype(np.float32), list(names) + list(en), list(kinds) + list(ek)

