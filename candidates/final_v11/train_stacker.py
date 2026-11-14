#!/usr/bin/env python3
"""Train v11's stacker (stacker_v11.txt): the S2 recipe of candidates/stacker_met/train_stacker.py, unchanged, on the
Metrica outputs of v11's four members.

    MTAH_WORK=... python -I candidates/final_v11/train_stacker.py [--out DIR]     # DIR default: this folder

Rows = frames t % 5 == 0 inside the graded blocks with a strict Metrica-rule label H / A / D, Sample Games 1, 2, 3 x bench
seeds 2027 and 2028, plus a home/away-swapped copy of every row. Inputs = stacker.feats (84 = 4 members x 7 offsets x
(H, A, D)) of the members' saved bench outputs met_<game>_<seed>.npy (float32) under $MTAH_WORK:
    TCN      pod_rescue/root/acc_ens/mw/probs/accens/tcn_tta   tcn_as (full model; never trained on Metrica), TTA4 + all_a50
    MS-TCN   pod_rescue/root/acc_ens/mw/probs/accens/ms_metft  ms_as_skcft_metft, LEAVE-ONE-GAME-OUT: game k's outputs come
                                                               from fold_met_k, which never saw game k (the shipped
                                                               full model trained on all three games, so its own Metrica
                                                               outputs would be in-sample and are not used)
    radar    probs/met_ft/parent_radar_pff                     radar_pff
    pointer  probs/met_ft/parent_p20M_pod                      r5_ptr_ballc_p20M (float32 GPU forward pass)
Model: lightgbm.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, objective="multiclass", random_state=0,
deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1). Writes stacker_v11.txt and stacker_v11_meta.json.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "stacker_met"), CAND, os.path.join(CAND, "mbr_decode")]
sys.path.append(os.path.join(CAND, "control_events"))
import stacker        # noqa: E402
import bench          # noqa: E402
import metrica_rules as mr   # noqa: E402

tsd = bench.tsd
WORK = os.environ["MTAH_WORK"]
ACC = os.path.join(WORK, "pod_rescue", "root", "acc_ens", "mw", "probs", "accens")
GAMES = ["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"]
SEEDS = (2027, 2028)
MET_DIRS = [os.path.join(ACC, "tcn_tta"), os.path.join(ACC, "ms_metft"),
            os.path.join(WORK, "probs", "met_ft", "parent_radar_pff"), os.path.join(WORK, "probs", "met_ft", "parent_p20M_pod")]
PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, objective="multiclass", random_state=0,
              deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1)


def strict_labels(g):
    folder = os.path.join(bench.SAMPLE_DIR, g)
    game = tsd.load_epts_game(folder) if g == "Sample_Game_3" else tsd.load_csv_game(folder)
    prac, blocks = tsd.practice_labels(game)
    return mr.strict_labels(game, prac, blocks)


def train_rows(games):
    X, y = [], []
    for g in games:
        strict = strict_labels(g)
        for seed in SEEDS:
            blocks = bench.prepared(g, seed)[4]
            P = [np.load(os.path.join(d, "met_{}_{}.npy".format(g, seed))).astype(np.float64) for d in MET_DIRS]
            runs = stacker.runs_of(blocks)
            n = len(P[0])
            assert n == len(strict)
            r = np.where(stacker.inside(n, runs) & (np.arange(n) % 5 == 0) & (strict < 3))[0]
            lab = strict[r].astype(np.int64)
            X += [stacker.feats(P, runs, r), stacker.feats(P, runs, r, swap=True)]
            y += [lab, np.where(lab == 0, 1, np.where(lab == 1, 0, lab))]
    return np.concatenate(X), np.concatenate(y)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    out = ap.parse_args().out
    X, y = train_rows(GAMES)
    m = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    path = os.path.join(out, "stacker_v11.txt")
    m.booster_.save_model(path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    imp = m.booster_.feature_importance("gain").reshape(4, len(stacker.OFFS), 3)
    meta = dict(lgb_version=lgb.__version__, sha256=sha, params=PARAMS, rows=int(len(y)),
                class_counts=np.bincount(y).tolist(), offsets_frames=stacker.OFFS.tolist(), fps=25,
                features="member-major (TCN, MS-TCN, radar, pointer), then offset, then class (H, A, D); 84",
                member_dirs_metrica=[os.path.relpath(d, WORK) for d in MET_DIRS], games=GAMES, bench_seeds=list(SEEDS),
                gain_share_member=np.round(imp.sum((1, 2)) / imp.sum(), 3).tolist(),
                gain_share_offset=np.round(imp.sum((0, 2)) / imp.sum(), 3).tolist())
    with open(os.path.join(out, "stacker_v11_meta.json"), "w") as h:
        json.dump(meta, h, indent=1)
        h.write("\n")
    print("trained on {} rows (class counts {}); wrote {} sha256 {}".format(len(y), meta["class_counts"], path, sha))
    print("gain share per member {} per offset {}".format(meta["gain_share_member"], meta["gain_share_offset"]))


if __name__ == "__main__":
    main()
