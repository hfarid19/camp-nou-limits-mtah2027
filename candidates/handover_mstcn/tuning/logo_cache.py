#!/usr/bin/env python3
"""Fast DFL leave-one-game-out loop (tuning only; the reported numbers come from bench.py).

    python logo_cache.py FOLDS_DIR OUT_DIR        # cache fold-model outputs per DFL game
    python logo_cache.py --sweep OUT_DIR 'json list of cfg dicts'

Uses the bench's own degradation of each game (bench.dfl_prepared, seed 2027) and the fold
model that never saw that game, exactly like bench --dfl, and caches features / TCN outputs.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.dirname(HERE)
sys.path.insert(0, CAND)
sys.path.insert(0, HERE)
import bench                     # noqa: E402
import possession_metrics as pm  # noqa: E402
import pipeline as P             # noqa: E402
import features as fe            # noqa: E402
import tcn_np                    # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

SCR = os.environ.get("HMT_SCRATCH", WORK + "/arch/handover_mstcn")
MATCHES = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))


def feats(match):
    path = os.path.join(SCR, "cache", "F_{}.npz".format(match))
    slots, slot_team, ball, gt, blocks, sign = bench.dfl_prepared(match, 2027)
    if not os.path.isfile(path):
        s = fe.frame_sign(len(ball), sign, blocks)
        F, names, kinds = fe.compute(slots, slot_team, ball, s, bench.tsd.PITCH, 25)
        np.savez(path, F=F, names=np.array(names), kinds=np.array(kinds))
    z = np.load(path)
    return z["F"], [str(x) for x in z["names"]], [str(x) for x in z["kinds"]], gt, blocks, ball


def run_model(model, F, names, kinds, blocks):
    perm, signs = fe.swap_index(names, kinds)
    n = len(F)
    pc = np.full((n, 3), 1.0 / 3.0)
    pb = np.zeros((n, 2))
    for _, first, last in blocks:
        for r in range(5):
            fr = np.arange(first + r, last + 1, 5)
            pc[fr], pb[fr] = model.predict(F[fr].astype(np.float64), perm, signs)
    return pc, pb


def cache(folds_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for gi, m in enumerate(MATCHES):
        F, names, kinds, gt, blocks, ball = feats(m)
        model = tcn_np.Model(os.path.join(folds_dir, "fold_tcn_{}.npz".format(gi)))
        pc, pb = run_model(model, F, names, kinds, blocks)
        np.savez(os.path.join(out_dir, "{}.npz".format(m)), pc=pc.astype(np.float32),
                 pb=pb.astype(np.float32), gt=gt, blocks=np.array(blocks))
        print(m, "cached", flush=True)


def sweep(out_dir, cfgs):
    data = []
    for m in MATCHES:
        z = np.load(os.path.join(out_dir, "{}.npz".format(m)))
        f = np.load(os.path.join(SCR, "cache", "F_{}.npz".format(m)))
        aux = P.post_aux(f["F"], [str(x) for x in f["names"]])
        data.append((m, z["pc"].astype(np.float64), z["pb"].astype(np.float64), z["gt"],
                     [tuple(int(v) for v in b) for b in z["blocks"]], aux))
    for cfg in cfgs:
        rows = []
        for m, pc, pb, gt, blocks, aux in data:
            pred = P.decode(pc, pb, blocks, len(gt), 25, cfg, aux)
            r = pm.evaluate(gt, pred, bootstrap_n=0)
            rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
        a = np.array(rows, float)
        print("spellF1 {:.4f} macroF1 {:.4f} poss {:.0f}/{:.0f}  {}".format(
            a[:, 0].mean(), a[:, 1].mean(), a[:, 2].sum(), a[:, 3].sum(), json.dumps(cfg)), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--sweep":
        sweep(sys.argv[2], json.loads(sys.argv[3]))
    else:
        cache(sys.argv[1], sys.argv[2])
