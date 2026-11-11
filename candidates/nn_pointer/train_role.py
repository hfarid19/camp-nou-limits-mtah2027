#!/usr/bin/env python3
"""Train r5_ptr_role (role-aware ball-centred pointer): train_ptr.py unchanged (same network ptr_net.PointerNet, loss,
recipe, folds, export) with two substitutions, so the only difference from r5_ptr_ballc is the input features:
  - inputs from <ds>/../ptrr (prep_role.py: role.prepare = ballc.prepare + 12 role / 12 shape columns)
  - team-swap augmentation = role.swap (ballc's negations + exchange of the home / away shape blocks)
cfg gains role=1 (stored in the exported npz; pipeline_role.py asserts it). Use with ballc=1, FS=22, FF=26, Lp=0:

    python -I candidates/nn_pointer/train_role.py --device cuda --ds $MTAH_WORK/ds_pff2/ds --out <dir> \
        --specs pff0,pff1,pff2,pff3 --cfg '{"ballc":1,"role":1,"FS":22,"FF":26,"Lp":0}'
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(HERE), "ball_radar_pff"))
import train_ptr as TP  # noqa: E402
import role             # noqa: E402

TP.CFG = dict(TP.CFG, role=0)
_make_batch = TP.make_batch
D = TP.D


def load(ds_dir, spec, ballc=False):
    """= train_ptr.load (ballc branch) with the slot inputs read from <ds>/../ptrr instead of ptrb."""
    assert ballc
    import glob
    hold = D.holdout()
    if D.allow_holdout():
        if spec.startswith("pff"):
            raise ValueError("NN_PFF_ALLOW_HOLDOUT=1: only full / dfl<g>")
        hold = set()
    shold = D.skc_holdout()
    ex = D.excluded(spec)
    base = os.path.dirname(os.path.abspath(ds_dir))
    seqs, games, names, kinds = [], set(), None, None
    for p in sorted(glob.glob(os.path.join(ds_dir, "*_*.npz"))):
        g = D.game_of(p)
        assert g[4:] not in hold or not g.startswith("PFF-"), p
        assert g not in shold, p
        if g in ex:
            continue
        bn = os.path.basename(p)
        y, bha, bd, yc, idx, half = D.targets(p, os.path.join(base, "raw", bn), os.path.join(base, "tgt"))
        z = np.load(p)
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
        pz = np.load(os.path.join(base, "ptrr", bn))
        oz = np.load(os.path.join(base, "owner", bn))
        assert np.array_equal(pz["idx"], idx) and np.array_equal(oz["idx"], idx), p
        X, xs, ms, xf, own = z["X"], pz["xs"], pz["ms"], pz["xf"], oz["own"]
        assert xs.shape[-1] == role.FS_R and xf.shape[-1] == role.FF_R, p
        own = TP.remap_owner(own.astype(np.int64), pz["sel"].astype(np.int64))
        brk = np.concatenate(([True], (np.diff(idx) != 5) | (np.diff(half) != 0)))
        sid = np.cumsum(brk) - 1
        for s in np.unique(sid):
            m = sid == s
            if m.sum() < 64:
                continue
            seqs.append(dict(g=g, X=X[m].astype(np.float32), y=y[m].astype(np.int64), yc=yc[m].astype(np.float32),
                             xs=xs[m], ms=ms[m], xf=xf[m], own=own[m].astype(np.int64)))
        games.add(g)
    return seqs, names, kinds, sorted(games)


def make_batch(data, w, rng, cfg, perm, signs):
    """TP.make_batch, then the shape blocks of the swapped items exchanged. TP.make_batch's ballc swap negates xf
    columns 0, 1, 10, 11 of a swapped item; the role swap additionally exchanges xf[14:20] <-> xf[20:26]. Swapped
    items are recognised by replaying the rng draws (same generator state, same draws)."""
    state = rng.bit_generator.state
    out = _make_batch(data, w, rng, cfg, perm, signs)
    if not cfg.get("role", 0):
        return out
    end = rng.bit_generator.state
    rng.bit_generator.state = state
    swapped = []
    for _ in range(cfg["batch"]):
        d = data[int(rng.choice(len(data), p=w))]
        n = len(d["y"])
        rng.integers(0, max(1, n - cfg["crop"]))
        swapped.append(rng.random() < 0.5)
    assert rng.bit_generator.state == end
    a, b = role.ballc.FF_B, role.ballc.FF_B + role.NG
    xf = out["xf"]
    for i, s in enumerate(swapped):
        if s:
            xf[i, :, a:a + role.NG], xf[i, :, b:b + role.NG] = xf[i, :, b:b + role.NG].copy(), xf[i, :, a:a + role.NG].copy()
    return out


TP.load = load
TP.make_batch = make_batch

if __name__ == "__main__":
    TP.main()
