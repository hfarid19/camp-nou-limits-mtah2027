#!/usr/bin/env python3
"""Idea 1 runner: per-game feature re-normalisation of the NN members on every bench unit.

    python candidates/adapt/run_renorm.py --shard 0 --nshard 4 [--domains dfl,skc,met] [--cfgs ...]
        [--variants base,all_a50,...]

Units: DFL bench v2 (7 games x seeds 2027-2029; the leave-one-game-out fold model fold_tcn_<gi>.npz of
each config, checked against its train_games), SkillCorner A-League (20 games, seed 2027, full model),
Metrica dev (Sample Games 1, 2 x seeds 2027, 2028, full model). One feature pass per unit; for every
config x variant: the member's own evidence with re-normalised inputs, then its own decoder.
Codes -> $MTAH_WORK/probs/adapt/<cfg>__<variant>/<tag>_codes.npy (tag as the benches: dfl2_<m>_<s>,
skc2_<g>_2027, dev_<g>_<s>); class probs (float16) next to them as <tag>.npy.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [HERE, CAND, os.path.join(CAND, "ensemble_ctrl_v6"), os.path.join(CAND, "skc"), ROOT,
                os.path.join(ROOT, "evaluation")]
WORK = os.environ["MTAH_WORK"]
ARCH = os.path.join(WORK, "arch", "nn_pff")
OUT = os.path.join(WORK, "probs", "adapt")

import renorm                     # noqa: E402
import members                    # noqa: E402

KIND = {"t5c": "tcn", "o_ls05": "tcn", "m5c": "mstcn", "m5c_W": "mstcn"}
_MOD = {}


def mods():
    if not _MOD:
        _MOD["tcn"] = members.load("nn_pff_tcn", "pipeline.py")
        _MOD["mstcn"] = members.load("nn_pff", "pipeline.py")
    return _MOD["tcn"], _MOD["mstcn"]


def units(domains):
    import bench
    out = []
    if "dfl" in domains:
        ms = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
        for gi, m in enumerate(ms):
            for s in bench.DFL2_SEEDS:
                out.append(("dfl", (m, s, gi), "dfl2_{}_{}".format(m, s)))
    if "skc" in domains:
        import bench_skc
        for g in bench_skc.all_games():
            out.append(("skc", (g,), "skc2_{}_2027".format(g)))
    if "met" in domains:
        for g in bench.DEV:
            for s in bench.SEEDS:
                out.append(("met", (g, s), "dev_{}_{}".format(g, s)))
    return out


def inputs(dom, u):
    import bench
    import tune_on_sample_data as tsd
    if dom == "dfl":
        z = bench.dfl2_prepared(u[0], u[1])
    elif dom == "skc":
        import bench_skc
        z = bench_skc.skc2_prepared(u[0])
    else:
        z = bench.prepared(u[0], u[1])
    slots, slot_team, ball, gt, blocks, sign = z
    return (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)


def weight_path(cfg, dom, u):
    if dom != "dfl":
        return os.path.join(ARCH, cfg, "full.npz")
    p = os.path.join(ARCH, cfg, "fold_tcn_{}.npz".format(u[2]))
    tg = {str(g) for g in np.load(p)["train_games"]}
    if u[0] in tg:
        raise RuntimeError("{} was trained on {}".format(p, u[0]))
    return p


def run_tcn(tmod, path, F, names, kinds, perm, mom, variants, args):
    players, _, ball, _, blocks, tags, nteams, _, fps = args
    W = tmod.tcn_np.load(path)
    out = {}
    for v in variants:
        mu, sd, m = renorm.variant_stats(v, W["mu"], W["sd"], mom, names, perm)
        W2 = dict(W)
        W2["mu"] = mu.astype(W["mu"].dtype) if v != "base" else W["mu"]
        W2["sd"] = sd.astype(W["sd"].dtype) if v != "base" else W["sd"]
        P, pc = tmod.tcn_probs(F, names, kinds, blocks, W2)
        codes = tmod.decode_probs(P, pc, F, names, float(W["change_prior"]), players, ball, blocks, tags,
                                  nteams, fps)
        out[v] = (np.asarray(codes, np.int8), P, int(m.sum()))
    return out


def run_mstcn(mmod, path, F, names, perm, signs, mom, variants, args):
    _, _, ball, _, blocks, _, _, _, fps = args
    model = mmod.tcn_np.Model(path)
    assert names == model.names
    orig = [(s.mean.copy(), s.std.copy()) for s in model.members]
    n = len(ball)
    aux = mmod.post_aux(F, names)
    out = {}
    for v in variants:
        nm = 0
        for s, (mu0, sd0) in zip(model.members, orig):
            mu, sd, m = renorm.variant_stats(v, mu0, sd0, mom, names, perm)
            s.mean, s.std = (mu0, sd0) if v == "base" else (mu, sd)
            nm = int(m.sum())
        pc = np.full((n, 3), 1.0 / 3.0)
        pb = np.zeros((n, 2))
        for _, first, last in blocks:
            for r in range(mmod.STRIDE):
                fr = np.arange(first + r, last + 1, mmod.STRIDE)
                if len(fr):
                    pc[fr], pb[fr] = model.predict(F[fr].astype(np.float64), perm, signs)
        codes = mmod.decode(pc, pb, blocks, n, fps, aux=aux)
        out[v] = (np.asarray(codes, np.int8), pc, nm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--domains", default="dfl,skc,met")
    ap.add_argument("--cfgs", default="o_ls05,m5c_W,t5c,m5c")
    ap.add_argument("--variants", default=",".join(renorm.VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="output root (default probs/adapt)")
    ap.add_argument("--p32", action="store_true", help="save class probs as float32 (MBR input)")
    a = ap.parse_args()
    global OUT
    OUT = a.out or OUT
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS", "NN_PFF_FOLD"):
        if k in os.environ:
            raise SystemExit(k + " must not be set")
    cfgs, variants = a.cfgs.split(","), a.variants.split(",")
    tmod, mmod = mods()
    fe = tmod.fe
    U = units(a.domains.split(","))[a.shard::a.nshard]
    if a.limit:
        U = U[:a.limit]
    for dom, u, tag in U:
        todo = [(c, v) for c in cfgs for v in variants
                if not os.path.isfile(os.path.join(OUT, "{}__{}".format(c, v), tag + "_codes.npy"))]
        if not todo:
            continue
        t0 = time.time()
        args = inputs(dom, u)
        players, slot_team, ball, sign_half1, blocks, tags, nteams, pitch, fps = args
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        perm, signs = fe.swap_index(names, kinds)
        mom = renorm.game_moments(F, blocks, perm, signs)
        t1 = time.time()
        for c in cfgs:
            vs = [v for v in variants if (c, v) in todo]
            if not vs:
                continue
            path = weight_path(c, dom, u)
            if KIND[c] == "tcn":
                res = run_tcn(tmod, path, F, names, kinds, perm, mom, vs, args)
            else:
                res = run_mstcn(mmod, path, F, list(names), perm, signs, mom, vs, args)
            for v, (codes, P, nm) in res.items():
                d = os.path.join(OUT, "{}__{}".format(c, v))
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, tag + ".npy"), P.astype(np.float32 if a.p32 else np.float16))
                np.save(os.path.join(d, tag + "_codes.npy"), codes)
        print("{} feat {:.0f}s total {:.0f}s".format(tag, t1 - t0, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
