#!/usr/bin/env python3
"""Match-specific (transductive) noisy student for BAR-ATH: MAC ONLY (the match's tracking never leaves it).
Never run, not shipped: the bench proxies of this idea were neutral (README.md), so no model was ever
fine-tuned on the match and nothing from this script is in the submission.

    python candidates/adapt/adapt_match.py build [--teacher lsw]       # pseudo-labels from the match's positions
    python candidates/adapt/adapt_match.py train --kind tcn|mstcn [--device mps|cpu]

Same recipe as the proxy runs on the benches (build_pl.py bench + train_student.py, jobs/transductive.txt):
  build  runs the teacher pipeline (lsw: ensemble_ctrl_v6/pipeline_lsw.py's members + v6 rule, prekick_dead)
         on the competition input exactly as make_submission.py reads it, computes the members' features at
         the five stride-5 phases over the graded blocks and writes them with the teacher's codes as
         $MTAH_WORK/arch/r5_adapt/pl_match/BAR-ATH/{ds,raw}/BAR-ATH_p<r>.npz. No labels, no evaluation/gt.
  train  fine-tunes the shipped full model (nn_pff_tcn/tcn_ls.npz or nn_pff/mstcn_w.npz, spec "full") on the
         labelled ds_pff2 data + the match's pseudo-labels (x8), fixed seed 0, fixed config (TCN: 600 its,
         lr 5e-4, ls 0.05; MS-TCN: 6 epochs, lr 2e-4), and exports numpy weights to
         $MTAH_WORK/arch/r5_adapt/match_student/{tcn_ls_match,mstcn_w_match}.npz.
Shipping: torch training (MPS / CPU) is not bit-reproducible, so make_submission.py must LOAD the exported
npz (committed next to the members' weights) rather than retrain; this script is the documented way to
regenerate them (same seed and config; outputs agree up to float noise, not byte for byte). The pipeline
then runs the members with those weights (the member files' WEIGHTS / MODEL_PATH), numpy only.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
WORK = os.environ["MTAH_WORK"]
OUT = os.path.join(WORK, "arch", "r5_adapt")
PL = os.path.join(OUT, "pl_match", "BAR-ATH")
CFG = {"tcn": dict(ls=0.05, iters=600, lr=5e-4, seed=0),
       "mstcn": dict(cosine=0, clip=0, C=128, Cr=64, epochs=6, lr=2e-4, seed=0)}
INIT = {"tcn": os.path.join(CAND, "nn_pff_tcn", "tcn_ls.npz"), "mstcn": os.path.join(CAND, "nn_pff", "mstcn_w.npz")}
NAME = {"tcn": "tcn_ls_match.npz", "mstcn": "mstcn_w_match.npz"}


def build(teacher):
    sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(CAND, "ensemble_ctrl_v6"), HERE]
    import possession_pipeline as pp
    import members
    import v6_core
    import ensemble_core as ec
    import build_pl
    import renorm
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    assert teacher == "lsw"
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    fps = GAMES[GAME]["fps"]
    args = (players, slot_team, ball, sign_half1, blocks, tags, names, GAMES[GAME]["pitch"], fps)
    vote = v6_core.Vote([("nn_pff_tcn", "pipeline_ls.py"), ("nn_pff", "pipeline_w.py"),
                         ("ball_radar_pff", "pipeline.py")], 1.0, 1.25)
    codes = np.asarray(vote.run(args)[0], np.int8)
    fe = vote.members[0].mod.fe
    sign = fe.frame_sign(n, sign_half1, blocks)
    F, fnames, kinds = fe.compute(players, slot_team, ball, sign, GAMES[GAME]["pitch"], fps)
    mask = renorm.block_rows(n, blocks)
    tc = np.where(mask, codes, 3).astype(np.int8)
    assert (tc[mask] < 3).all()
    for r in range(5):
        idx = np.where(mask & (np.arange(n) % 5 == r))[0]
        build_pl.write(PL, "BAR-ATH_p{}".format(r), F[idx], tc, idx, blocks, fnames, kinds)
    print("pseudo-labels -> {} ({} graded frames, teacher {})".format(PL, int(mask.sum()), teacher))


def train(kind, device):
    out = os.path.join(OUT, "match_student")
    cfg = dict(CFG[kind], init=INIT[kind])
    cmd = [sys.executable, "-I", os.path.join(HERE, "train_student.py"), "--kind", kind, "--device", device,
           "--ds", os.path.join(WORK, "ds_pff2", "ds"), "--pl", PL + ":8", "--specs", "full", "--out", out,
           "--name", NAME[kind], "--cfg", json.dumps(cfg)]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", OMP_NUM_THREADS="2")
    subprocess.check_call(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "train"])
    ap.add_argument("--teacher", default="lsw")
    ap.add_argument("--kind", default="tcn", choices=["tcn", "mstcn"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    if a.step == "build":
        build(a.teacher)
    else:
        train(a.kind, a.device)


if __name__ == "__main__":
    main()
