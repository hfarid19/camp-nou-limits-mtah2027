"""Drop-in replacements for the `fe` module of nn_pff_tcn/pipeline.py and nn_pff/pipeline.py for the v12 members: the repo
features (candidates/nn_pff/features.py, unchanged) with extra column blocks appended in the fixed order
ACC, SHAPE, REL, KIN2, INT (each present only if the member's group list has it), float32. swap_index / frame_sign are the
repo's. The blocks come from this folder's nf_features.extra (ACC + SHAPE, 29 columns), nf_rel.rel (19), nf_kin2.kin2 (26)
and nf_int.inter (13): the round-5 "newfeat" experiment's code (docs/prereg/round5_prereg.txt addenda of 13 Nov 2026).

make(groups) -> an object with compute / swap_index / frame_sign. Every instance shares one in-memory cache of the blocks
of the last few inputs (players, slot_team, ball, pitch, fps; none of the blocks depends on the attack sign), so the
MS-TCN member reuses the passes the TCN member already made on the unjittered input. A cache hit returns the array that was
computed, so the output does not depend on the cache. Nothing is written to disk.
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(fname, name):
    if name not in sys.modules:
        sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
    return sys.modules[name]


nf = _load("nf_features.py", "mtah_newfeat_v12_nf_features")
nf_rel = _load("nf_rel.py", "mtah_newfeat_v12_nf_rel")
nf_kin2 = _load("nf_kin2.py", "mtah_newfeat_v12_nf_kin2")
nf_int = _load("nf_int.py", "mtah_newfeat_v12_nf_int")
swap_index = nf.fe.swap_index
frame_sign = nf.fe.frame_sign
ORDER = ("acc", "shape", "rel", "kin2", "int")
WIDTH = {"as": 29, "rel": 19, "kin2": 26, "int": 13}
KEEP = 6                 # inputs kept (one TTA member pass = the original + 4 jittered copies)
_CACHE = {}              # key -> {block: (E, names, kinds)}, insertion-ordered


def _key(players, slot_team, ball, pitch, fps):
    h = hashlib.sha1()
    for a in (players, slot_team, ball):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    h.update(repr((tuple(pitch), fps)).encode())
    return h.hexdigest()


def block(name, players, slot_team, ball, pitch, fps):
    k = _key(players, slot_team, ball, pitch, fps)
    if k not in _CACHE:
        while len(_CACHE) >= KEEP:
            del _CACHE[next(iter(_CACHE))]
        _CACHE[k] = {}
    c = _CACHE[k]
    if name not in c:
        if name == "as":
            c[name] = nf.extra(players, slot_team, ball, pitch, fps, nf.GROUPS)
        elif name == "rel":
            c[name] = nf_rel.rel(players, slot_team, ball, pitch, fps)
        elif name == "kin2":
            c[name] = nf_kin2.kin2(players, slot_team, ball, pitch, fps)
        elif name == "int":
            c[name] = nf_int.inter(players, slot_team, ball, pitch, fps)
        assert c[name][0].shape[1] == WIDTH[name], (name, c[name][0].shape)
    return c[name]


class _Fe(object):
    def __init__(self, groups):
        groups = tuple(groups)
        assert set(groups) <= set(ORDER) and groups, groups
        self.groups = tuple(g for g in ORDER if g in groups)
        self.swap_index = swap_index
        self.frame_sign = frame_sign

    def compute(self, players, slot_team, ball, sign, pitch, fps):
        F, names, kinds = nf.fe.compute(players, slot_team, ball, sign, pitch, fps)
        cols, names, kinds = [F], list(names), list(kinds)
        if "acc" in self.groups or "shape" in self.groups:
            E, en, ek = block("as", players, slot_team, ball, pitch, fps)
            acc = set(en[:14])
            keep = [i for i, nm in enumerate(en) if (nm in acc and "acc" in self.groups) or
                    (nm not in acc and "shape" in self.groups)]
            cols.append(E[:, keep])
            names += [en[i] for i in keep]
            kinds += [ek[i] for i in keep]
        for g in ("rel", "kin2", "int"):
            if g in self.groups:
                E, en, ek = block(g, players, slot_team, ball, pitch, fps)
                cols.append(E)
                names += list(en)
                kinds += list(ek)
        return np.concatenate(cols, 1).astype(np.float32), names, kinds


_FES = {}


def make(groups):
    key = tuple(g for g in ORDER if g in groups)
    if key not in _FES:
        _FES[key] = _Fe(key)
    return _FES[key]
