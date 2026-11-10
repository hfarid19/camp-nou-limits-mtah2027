#!/usr/bin/env python3
"""Noisy student / distillation trainer (idea 3): the nn_pff TCN or MS-TCN recipe, trained on the labelled
data (data.py specs) PLUS pseudo-labelled files (build_pl.py), unchanged trainers otherwise.

    python candidates/adapt/train_student.py --kind tcn|mstcn --ds $W/ds_pff2/ds --pl DIR[:REP] [--pl ...]
        --out OUT --specs dfl0,...,dfl6,skc0,skc1,skc2,full [--cfg '{...}'] [--device cuda]
        [--only-pl GAME,...]

Specs: dfl<g> / full / pff<k> as data.py for the labelled part, every pseudo-labelled game included;
skc<j>: the labelled `full` set + the pseudo-labelled games EXCEPT SkillCorner fold j (SKC_FOLDS: the 15
SKC training games in 3 fixed folds of 5), so fold_skc_<j>.npz never saw (even pseudo-labels of) the SKC
games it scores. Pseudo-labelled sequences are named "PL:<game>" in train_games and repeated REP times
(sampling weight; they sit in the non-DFL pool of the recipes' w_dfl split).
Transductive (match-style) fine-tune: --cfg '{"init": <the model to adapt>}' --specs <its spec>
--pl <one unit's files>. Output names: as the recipes (fold_tcn_<g>.npz, full.npz) + fold_skc_<j>.npz.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
import torch  # noqa: E402,F401  (before nn_pff on the path: its queue.py shadows the stdlib's)
sys.path.append(os.path.join(CAND, "nn_pff"))
import data as D  # noqa: E402

SKC_TRAIN = ["SKC-1886347", "SKC-1925299", "SKC-1927964", "SKC-1959846", "SKC-1996435", "SKC-1996436",
             "SKC-2006363", "SKC-2007448", "SKC-2007721", "SKC-2010085", "SKC-2011166", "SKC-2013725",
             "SKC-2015213", "SKC-2016236", "SKC-2017461"]
_perm = np.random.default_rng(20261109).permutation(len(SKC_TRAIN))
SKC_FOLDS = [sorted(SKC_TRAIN[i] for i in _perm[j::3]) for j in range(3)]

PL = []           # [(dir, rep)]
ONLY = []         # restrict pseudo-labelled games
_orig_load = D.load
_orig_excluded = D.excluded


def excluded(spec):
    if spec.startswith("skc"):
        return set()
    return _orig_excluded(spec)


def load(ds_dir, spec, *a, **k):
    base = "full" if spec.startswith("skc") else spec
    seqs, names, kinds, games = _orig_load(ds_dir, base, *a, **k)
    drop = set(SKC_FOLDS[int(spec[3:])]) if spec.startswith("skc") else set()
    games = list(games)
    for d, rep in PL:
        ps, n2, k2, g2 = _orig_load(d, "full")
        assert ps, "no pseudo-labelled sequences in " + d
        assert n2 == names and k2 == kinds, "pseudo-label features differ"
        keep = [("PL:" + g, X, y, b) for g, X, y, b in ps if g not in drop and (not ONLY or g in ONLY)]
        seqs = seqs + keep * rep
        games += sorted({q[0] for q in keep})
    return seqs, names, kinds, sorted(set(games))


D.load = load
D.excluded = excluded


def out_name(out, spec):
    if spec.startswith("skc"):
        return os.path.join(out, "fold_skc_{}.npz".format(int(spec[3:])))
    if spec == "full":
        return os.path.join(out, "full.npz")
    if spec.startswith("pff"):
        return os.path.join(out, "fold_pff_{}.npz".format(int(spec[3:])))
    return os.path.join(out, "fold_tcn_{}.npz".format(int(spec[3:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["tcn", "mstcn"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--pl", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only-pl", default="")
    ap.add_argument("--name", default="", help="output file name (single spec; transductive runs)")
    a = ap.parse_args()
    for p in a.pl:
        d, _, rep = p.partition(":")
        if os.path.isdir(os.path.join(d, "ds")):
            d = os.path.join(d, "ds")
        PL.append((d, int(rep or 1)))
    ONLY.extend([g for g in a.only_pl.split(",") if g])
    if a.kind == "tcn":
        import train_tcn as T
    else:
        import train_mstcn as T
    assert T.D is D
    cfg = dict(T.CFG, **json.loads(a.cfg))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_cfg.json"), "w") as h:
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), pl=PL, only_pl=ONLY, model=a.kind,
                       skc_folds=SKC_FOLDS), h, indent=1)
    logf = open(os.path.join(a.out, "train_log.txt"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    log("cfg " + json.dumps(cfg) + " specs " + a.specs + " pl " + json.dumps(PL))
    for spec in a.specs.split(","):
        p = os.path.join(a.out, a.name) if a.name else out_name(a.out, spec)
        if os.path.isfile(p):
            log("{} exists, skipped".format(p))
            continue
        T.train_spec(a.ds, spec, p, cfg, a.threads, log, a.device)


if __name__ == "__main__":
    main()
