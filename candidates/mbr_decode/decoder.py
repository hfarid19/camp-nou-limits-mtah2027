"""The MBR decoder on one game: candidates (rule variants of the hard vote and of each member), posterior
samples (HMM on the pooled member probabilities, or the members' own codes), window selection (mbr.py).

decode(probs, codes, blocks, cfg) -> (codes, info); cfg keys:
    src     "hmm": N_SAMPLES FFBS paths of mbr.ffbs on the arithmetic mean of the members' probabilities
                   (alpha = temperature, rho = exit probability per STRIDE frames, R sub-states per label)
            "mem": the members' own decoded codes
            "mix": both, each source weighted 1/2
    margin  a non-base candidate must beat the base by more than this (objective units: 2 per fully
            matched spell)
Every sample is read at the three granularities of the benches' fine targets: control.fold at T = 0, 0.5
and 1 s (raw flag, 0.5 s and 1 s folds), weighted equally.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), CAND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import control              # noqa: E402
import ensemble_core as ec  # noqa: E402
import members              # noqa: E402
import mbr                  # noqa: E402

FPS = 25
N_SAMPLES = 32
STRIDE = 5
R = 4
SEED = 2027
SAMPLE_FOLDS = (0.0, 0.5, 1.0)
GRID = [(s, T) for s in (0.0, 1.0) for T in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)]
BASE = (1.0, 1.25)
_LDE = []


def lde():
    if not _LDE:
        _LDE.append(members.load("learned_dfl_evidence"))
    return _LDE[0]


def rule(codes_list, blocks, smooth, T):
    """ensemble_ctrl_v6's vote machinery (= vote_offline.vote): hard vote -> merge_short_v2 -> fold."""
    sm = dict(min_poss_s=smooth, min_dead_s=smooth) if smooth > 0 else None
    if len(codes_list) == 1 and sm is None:
        v = np.asarray(codes_list[0], np.int8).copy()
    else:
        v = ec.hard_vote(codes_list, blocks, lde(), FPS, None, sm)
    return control.fold(np.asarray(v, np.int8), T, mbr.runs_of(blocks), FPS)


def base(codes, blocks):
    return rule(codes, blocks, *BASE)


def candidates(codes, blocks):
    C = [base(codes, blocks)]
    for s, T in GRID:
        if (s, T) != BASE:
            C.append(rule(codes, blocks, s, T))
    for c in codes:
        C.append(rule([c], blocks, 0.0, 0.0))
        C.append(rule([c], blocks, *BASE))
    out = []
    for c in C:
        if not any(np.array_equal(c, o) for o in out):
            out.append(c)
    return out


def fold_all(paths, blocks, folds=SAMPLE_FOLDS):
    runs = mbr.runs_of(blocks)
    return [control.fold(np.asarray(p, np.int8), T, runs, FPS) for p in paths for T in folds]


def samples(probs, codes, blocks, cfg):
    n = len(codes[0])
    out, w = [], []
    if cfg["src"] in ("hmm", "mix"):
        P = np.mean(probs, 0)
        S = fold_all(mbr.sample_paths(P, blocks, n, cfg["alpha"], cfg["rho"], cfg.get("r", R), STRIDE,
                                      cfg.get("n", N_SAMPLES), cfg.get("seed", SEED)), blocks,
                     cfg.get("folds", SAMPLE_FOLDS))
        out += S
        w += [1.0 / len(S)] * len(S)
    if cfg["src"] in ("mem", "mix"):
        S = fold_all(codes, blocks, cfg.get("folds", SAMPLE_FOLDS))
        out += S
        w += [1.0 / len(S)] * len(S)
    return out, np.asarray(w)


def decode(probs, codes, blocks, cfg, cache=None):
    cache = {} if cache is None else cache
    runs = mbr.runs_of(blocks)
    if "cands" not in cache:
        cache["cands"] = candidates(codes, blocks)
    C = cache["cands"]
    sk = tuple(sorted((k, v) for k, v in cfg.items() if k != "margin"))
    if sk not in cache:
        S, w = samples(probs, codes, blocks, cfg)
        cache[sk] = mbr.utilities(C, S, runs, w)
    return mbr.select_windows(C, None, runs, margin=cfg.get("margin", 0.0), util=cache[sk])
