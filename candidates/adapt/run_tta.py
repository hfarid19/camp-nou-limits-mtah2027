#!/usr/bin/env python3
"""Idea 2: test-time ensembling over mild input perturbations (on top of the members' mirror + 5-phase passes).

    python candidates/adapt/run_tta.py --shard i --nshard n [--cfgs o_ls05,m5c_W,t5c,m5c] [--k 4]

For every bench unit: the original pass + K jittered passes. Pass r (numpy default_rng(1000 + r), fixed):
every player slot and the ball get a smooth AR(1) offset (time constant 1 s; stationary sd 0.5 m for
players, 0.3 m for the ball: inside the training degradations' ball offset range), NaNs kept; features
recomputed; the member's class probs (TCN: + hand-over prob; MS-TCN: + boundary probs) averaged over the
K + 1 passes; then the member's own decoder (its raw-feature inputs = the original pass). Optionally
combined with a renorm variant (--renorm all_a50: each pass normalised with the ORIGINAL pass's game stats).
Codes -> $MTAH_WORK/probs/adapt/<cfg>__tta<K>[_<renorm>]/<tag>_codes.npy.
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

from tta import jitter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--domains", default="dfl,skc,met")
    ap.add_argument("--cfgs", default="o_ls05,m5c_W,t5c,m5c")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--renorm", default="base")
    ap.add_argument("--out", default=None, help="output root (default probs/adapt)")
    ap.add_argument("--p32", action="store_true", help="save class probs as float32 (MBR input)")
    a = ap.parse_args()
    if a.out:
        rr.OUT = a.out
    tmod, mmod = rr.mods()
    fe = tmod.fe
    cfgs = a.cfgs.split(",")
    vname = "tta{}".format(a.k) + ("" if a.renorm == "base" else "_" + a.renorm)
    for dom, u, tag in rr.units(a.domains.split(","))[a.shard::a.nshard]:
        todo = [c for c in cfgs if not os.path.isfile(os.path.join(rr.OUT, "{}__{}".format(c, vname), tag + "_codes.npy"))]
        if not todo:
            continue
        t0 = time.time()
        args = rr.inputs(dom, u)
        players, slot_team, ball, sign_half1, blocks, tags, nteams, pitch, fps = args
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F0, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        perm, signs = fe.swap_index(names, kinds)
        mom = renorm.game_moments(F0, blocks, perm, signs)
        Fs = [F0]
        for r in range(a.k):
            pj, bj = jitter(players, ball, r, fps)
            Fs.append(fe.compute(pj, slot_team, bj, sign, pitch, fps)[0])
        n = len(ball)
        for c in todo:
            path = rr.weight_path(c, dom, u)
            if rr.KIND[c] == "tcn":
                W = tmod.tcn_np.load(path)
                W2 = dict(W)
                if a.renorm != "base":
                    mu, sd, _ = renorm.variant_stats(a.renorm, W["mu"], W["sd"], mom, names, perm)
                    W2["mu"], W2["sd"] = mu.astype(W["mu"].dtype), sd.astype(W["sd"].dtype)
                outs = [tmod.tcn_probs(F, names, kinds, blocks, W2) for F in Fs]
                P = np.mean([o[0] for o in outs], 0)
                pc = np.mean([o[1] for o in outs], 0)
                codes = tmod.decode_probs(P, pc, F0, names, float(W["change_prior"]), players, ball, blocks,
                                          tags, nteams, fps)
            else:
                model = mmod.tcn_np.Model(path)
                if a.renorm != "base":
                    for s in model.members:
                        s.mean, s.std, _ = renorm.variant_stats(a.renorm, s.mean, s.std, mom, list(names), perm)
                Ps, Bs = [], []
                for F in Fs:
                    pcl = np.full((n, 3), 1.0 / 3.0)
                    pb = np.zeros((n, 2))
                    for _, first, last in blocks:
                        for r in range(mmod.STRIDE):
                            fr = np.arange(first + r, last + 1, mmod.STRIDE)
                            if len(fr):
                                pcl[fr], pb[fr] = model.predict(F[fr].astype(np.float64), perm, signs)
                    Ps.append(pcl)
                    Bs.append(pb)
                P = np.mean(Ps, 0)
                codes = mmod.decode(P, np.mean(Bs, 0), blocks, n, fps, aux=mmod.post_aux(F0, list(names)))
            d = os.path.join(rr.OUT, "{}__{}".format(c, vname))
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, tag + ".npy"), P.astype(np.float32 if a.p32 else np.float16))
            np.save(os.path.join(d, tag + "_codes.npy"), np.asarray(codes, np.int8))
        print("{} {:.0f}s".format(tag, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
