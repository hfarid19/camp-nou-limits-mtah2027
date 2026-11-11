"""nn_pointer, formation / role-aware variant r5_ptr_role (role.py: r5_ptr_ballc's inputs + per-player role features
from team shape + both teams' shape summaries; train_role.py cfg ballc=1, role=1, FS 22, FF 26, Lp 0), 48 PFF + 7 DFL
games (ds_pff2). pipeline.py executed under a private module name, its ptr_probs replaced by the role version below
(identical except role.prepare / role.swap), weights ptr_role.npz (loaded through pff_folds.resolve, so the DFL / PFF
benches use that file's fold models). Bench wrapper: NN_PFF_MODEL=<full.npz> overrides the weights (as
pipeline_cfg.py). Numpy only, float64, deterministic."""

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_pointer_variant_role", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.WEIGHTS = os.path.abspath(os.environ["NN_PFF_MODEL"]) if os.environ.get("NN_PFF_MODEL") else \
    os.path.join(HERE, "ptr_role.npz")

import role  # noqa: E402  (pipeline.py put HERE and ball_radar_pff on sys.path)

fe, ptr_np, STRIDE = _base.fe, _base.ptr_np, _base.STRIDE


def ptr_probs(players, slot_team, ball, sign_half1, blocks, F, names, kinds, W, meta):
    assert list(names) == meta["names"], "feature list differs from the model"
    cfg = meta["cfg"]
    assert cfg.get("role", 0) == 1 and cfg.get("ballc", 0) == 1 and cfg["FS"] == role.FS_R and cfg["FF"] == role.FF_R, cfg
    if _base.MIX >= 0:
        meta = dict(meta, cfg=dict(cfg, mix=_base.MIX))
    n = len(ball)
    sf = fe.frame_sign(n, sign_half1, blocks)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float64)
    Xn = np.nan_to_num((F.astype(np.float64) - meta["mean"]) / meta["std"])
    Xm = Xn[:, perm] * signs
    P = np.full((n, 3), 1.0 / 3.0)
    pc = np.zeros(n)
    for _, first, last in blocks:
        used = ~np.isnan(players[first:last + 1, :, 0]).all(0)
        pl, st = players[:, used], np.asarray(slot_team)[used]
        for p in range(STRIDE):
            if first + p > last:
                continue
            fr = np.arange(first + p, last + 1, STRIDE)
            inp = role.prepare(pl, st, ball, sf, 25.0, fr)
            # the training inputs are stored as float16 (prep_role.py); round the same way
            xs = inp["xs"].astype(np.float16).astype(np.float64)
            xf = inp["xf"].astype(np.float16).astype(np.float64)
            ms = inp["ms"].astype(np.float64)
            o = ptr_np.forward(W, meta, xs, ms, xf, Xn[fr])
            xs2, xf2 = role.swap(xs, xf)
            m = ptr_np.forward(W, meta, xs2, ms, xf2, Xm[fr])
            P[fr] = 0.5 * (o["P"] + m["P"][:, [1, 0, 2]])
            pc[fr] = 0.5 * (o["ho"] + m["ho"])
    return P, pc


_base.ptr_probs = ptr_probs
predict = _base.predict
predict_proba = _base.predict_proba
