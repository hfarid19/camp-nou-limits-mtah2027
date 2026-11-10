#!/usr/bin/env python3
"""Pseudo-labelled training files for the noisy student (idea 3), in the ds_pff format (data.py reads them).

    python candidates/adapt/build_pl.py skc  [--teacher lsw] [--out $MTAH_WORK/arch/r5_adapt/pl_skc_lsw]
    python candidates/adapt/build_pl.py bench --domain dfl|skc|met [--teacher lsw] --out DIR

skc    inductive: the 15 SkillCorner training games (never the 5 SKC holdout games), every degraded
       training copy in $MTAH_WORK/skc_aleague/ds (seeds 41 / 42 / 44, X / idx unchanged), labels
       replaced by the teacher's codes. Teacher = the vote of the teacher's members' saved codes on the
       game's bench input (skc2_<g>_2027) under the v6 rule (merge_short 1 s, fold 1.25 s): the same
       frame-level labels serve every degraded copy of the game. No SKC label is read (asserted: the
       teacher codes come only from probs/).
bench  transductive (the match-style version, proxy): each bench unit's OWN input (the frames the
       pipeline scores), features at the five stride-5 phases, labelled by the teacher's codes on that
       same input (DFL: the members' leave-one-game-out codes). One folder per unit, one file per phase:
       <out>/<tag>/ds/<tag>_p<r>.npz, <out>/<tag>/raw/<tag>_p<r>.npz (codes = teacher codes, blocks);
       data.game_of names all five phases "<tag>".
Files: ds/<name>_<seed>.npz (X, y, yc, idx, names, kinds), raw/<name>_<seed>.npz (codes, blocks).
"""

import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(CAND, "nn_pff")]
import score_adapt as sa     # noqa: E402
import vote_offline as vo    # noqa: E402

WORK = vo.WORK
TEACHERS = {"lsw": sa.LSW.split(";"), "v6": sa.V6.split(";")}


def teacher_codes(teacher, dom, tag, blocks):
    return vo.vote([sa.load(m, dom, tag) for m in TEACHERS[teacher]], blocks, *sa.RULE).astype(np.int8)


def change_target(codes, hw=4):
    c = np.zeros(len(codes), bool)
    t = np.where((codes[1:] != codes[:-1]) & (codes[1:] < 2) & (codes[:-1] < 2))[0] + 1
    for k in range(-hw, hw + 1):
        c[np.clip(t + k, 0, len(c) - 1)] = True
    return c.astype(np.float32)


def write(out, name, X, codes, idx, blocks, names, kinds):
    for sub in ("ds", "raw"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    np.savez_compressed(os.path.join(out, "ds", name + ".npz"), X=X.astype(np.float32), y=codes[idx], yc=change_target(codes)[idx],
             idx=idx, names=np.array(names), kinds=np.array(kinds))
    np.savez(os.path.join(out, "raw", name + ".npz"), codes=codes, blocks=np.array(blocks))


def build_skc(teacher, out):
    import json
    sys.path.insert(0, os.path.join(CAND, "skc"))
    import bench_skc
    hold = set(json.load(open(os.path.join(WORK, "skc_aleague", "holdout_games.json")))["holdout"])
    src = os.path.join(WORK, "skc_aleague", "ds")
    files = sorted(f for f in os.listdir(os.path.join(src, "ds")) if f.endswith(".npz"))
    agree = []
    for f in files:
        name = f[:-4]
        game = name.rsplit("_", 1)[0]
        assert game not in hold, game
        g = game[4:]
        blocks = bench_skc.skc2_prepared(g)[4]
        n = len(np.load(os.path.join(src, "raw", f))["codes"])     # length only
        tc = teacher_codes(teacher, "skc", "skc2_{}_2027".format(g), blocks)
        assert len(tc) == n
        z = np.load(os.path.join(src, "ds", f))
        idx = z["idx"].astype(np.int64)
        assert (tc[idx] < 3).all()
        write(out, name, z["X"], tc, idx, blocks, [str(s) for s in z["names"]], [str(s) for s in z["kinds"]])
        agree.append(name)
        print(name, len(idx), np.bincount(tc[idx], minlength=3), flush=True)
    print("{} files -> {}".format(len(agree), out))


def build_bench(teacher, dom, out, only=None):
    import run_renorm as rr
    tmod, _ = rr.mods()
    fe = tmod.fe
    for d, u, tag in rr.units([dom]):
        if only and tag not in only:
            continue
        if os.path.isfile(os.path.join(out, tag, "raw", tag + "_p4.npz")):
            continue
        args = rr.inputs(d, u)
        players, slot_team, ball, sign_half1, blocks, tags, nteams, pitch, fps = args
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        tc = teacher_codes(teacher, d, tag, blocks)
        mask = rr.renorm.block_rows(len(ball), blocks)
        for r in range(5):
            idx = np.where(mask & (np.arange(len(ball)) % 5 == r))[0]
            write(os.path.join(out, tag), "{}_p{}".format(tag, r), F[idx], tc, idx, blocks, names, kinds)
        print(tag, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["skc", "bench"])
    ap.add_argument("--teacher", default="lsw")
    ap.add_argument("--domain", default="skc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.mode == "skc":
        build_skc(a.teacher, a.out or os.path.join(WORK, "arch", "r5_adapt", "pl_skc_" + a.teacher))
    else:
        build_bench(a.teacher, a.domain, a.out or os.path.join(WORK, "arch", "r5_adapt", "pl_bench_{}_{}".format(a.domain, a.teacher)), a.only)


if __name__ == "__main__":
    main()
