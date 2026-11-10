#!/usr/bin/env python3
"""Bench codes of student models (idea 3) with the members' own numpy inference + decoder.

    python candidates/adapt/run_student.py --model NAME=tcn|mstcn:DIR [--model ...] [--shard i --nshard n]
        [--domains dfl,skc,met] [--variants base,...] [--trans]

Weights per unit (each checked against the file's train_games, pseudo-labels included as "PL:<game>"):
  DFL game gi   DIR/fold_tcn_<gi>.npz           (never saw the game, labelled or pseudo-labelled)
  SKC game      DIR/fold_skc_<j>.npz if the game is in SkillCorner fold j (train_student.SKC_FOLDS),
                else DIR/full.npz (the 5 SKC holdout games: never pseudo-labelled)
  Metrica dev   DIR/full.npz
--trans (transductive students): DIR/<tag>.npz, the model fine-tuned on that unit's own pseudo-labels
  (its train_games must contain "PL:<tag>" and no other pseudo-labelled or labelled copy of the game).
Codes -> $MTAH_WORK/probs/adapt/<NAME>[__<variant>]/<tag>_codes.npy.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_renorm as rr   # noqa: E402
import renorm             # noqa: E402

SKC_FOLDS = None


def skc_folds():
    global SKC_FOLDS
    if SKC_FOLDS is None:
        import json
        SKC_FOLDS = json.load(open(os.path.join(HERE, "skc_folds.json")))
    return SKC_FOLDS


def weight(d, dom, u, tag, trans):
    if trans:
        p = os.path.join(d, tag + ".npz")
    elif dom == "dfl":
        p = os.path.join(d, "fold_tcn_{}.npz".format(u[2]))
    elif dom == "skc":
        g = "SKC-" + u[0]
        j = [k for k, f in enumerate(skc_folds()) if g in f]
        p = os.path.join(d, "fold_skc_{}.npz".format(j[0]) if j else "full.npz")
        if j and not os.path.isfile(p):        # no pseudo-labels (control models): the full model
            p = os.path.join(d, "full.npz")
    else:
        p = os.path.join(d, "full.npz")
    tg = {str(g) for g in np.load(p)["train_games"]}
    game = u[0] if dom == "dfl" else ("SKC-" + u[0] if dom == "skc" else u[0])
    if game in tg or "PL:" + game in tg:
        raise RuntimeError("{} was trained on {}".format(p, game))
    if trans:
        assert "PL:" + tag in tg, p
        assert not any(t.startswith("PL:") and t != "PL:" + tag for t in tg), p
    else:
        assert not any(t.startswith(("PL:dfl2_", "PL:dev_", "PL:skc2_")) for t in tg), p
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--domains", default="dfl,skc,met")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--trans", action="store_true")
    a = ap.parse_args()
    models = []
    for m in a.model:
        name, spec = m.split("=", 1)
        kind, d = spec.split(":", 1)
        models.append((name, kind, d))
    variants = a.variants.split(",")
    tmod, mmod = rr.mods()
    fe = tmod.fe
    U = rr.units(a.domains.split(","))[a.shard::a.nshard]
    for dom, u, tag in U:
        def out(name, v):
            return os.path.join(rr.OUT, name if v == "base" else "{}__{}".format(name, v))
        todo = [(n, k, d) for n, k, d in models
                if any(not os.path.isfile(os.path.join(out(n, v), tag + "_codes.npy")) for v in variants)]
        todo = [(n, k, d) for n, k, d in todo if not a.trans or os.path.isfile(os.path.join(d, tag + ".npz"))]
        if not todo:
            continue
        t0 = time.time()
        args = rr.inputs(dom, u)
        players, slot_team, ball, sign_half1, blocks, tags, nteams, pitch, fps = args
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        perm, signs = fe.swap_index(names, kinds)
        mom = renorm.game_moments(F, blocks, perm, signs)
        for n, k, d in todo:
            p = weight(d, dom, u, tag, a.trans)
            if k == "tcn":
                res = rr.run_tcn(tmod, p, F, names, kinds, perm, mom, variants, args)
            else:
                res = rr.run_mstcn(mmod, p, F, list(names), perm, signs, mom, variants, args)
            for v, (codes, P, _) in res.items():
                o = out(n, v)
                os.makedirs(o, exist_ok=True)
                np.save(os.path.join(o, tag + ".npy"), P.astype(np.float16))
                np.save(os.path.join(o, tag + "_codes.npy"), codes)
        print("{} {:.0f}s".format(tag, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
