#!/usr/bin/env python3
"""Train the Metrica stacker S2 (stacker_s2.txt) from the four F4sP members' saved outputs on Metrica, and check it.

    MTAH_WORK=... python -I candidates/stacker_met/train_stacker.py train [--out DIR]   # DIR default: this folder
    MTAH_WORK=... python -I candidates/stacker_met/train_stacker.py check [--model FILE]

train: the recipe of arch/r5_arch/stack_aug/code/stack.py (kind "S2", trained on all three games), unchanged. Rows = frames
t % 5 == 0 inside the graded blocks with a strict Metrica-rule label H / A / D (candidates/control_events/metrica_rules.py),
for Sample Games 1, 2, 3 x bench seeds 2027 and 2028 (degraded inputs of candidates/bench.py), plus a home/away-swapped copy
of every row (H and A columns of every member and offset exchanged, label H <-> A). Inputs = stacker.feats of the members'
saved bench outputs $MTAH_WORK/probs/<dir>/met_<game>_<seed>.npy (float32), dirs:
    TCN      adapt32/o_ls05_ref__tta4_all_a50   (o_ls05, test-time averaging over 4 jittered copies, all_a50 renorm)
    MS-TCN   met_ft/parent_m5c_skcft            (m5c_C192ls_skcft, the shipped v8 model)
    radar    met_ft/parent_radar_pff            (radar_pff)
    pointer  met_ft/parent_p20M_pod             (r5_ptr_ballc_p20M; float32 GPU forward pass)
None of these members trained on Metrica. Model: lightgbm.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
objective="multiclass", random_state=0, deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1), other parameters
default. Writes stacker_s2.txt (booster text model) and stacker_s2_meta.json (lightgbm version, sha256, rows).

check: applies the given model file (default the shipped one) through stacker.predict and compares with the experiment's
saved outputs of its all-games S2 ($MTAH_WORK/probs/stack_aug/S2_full: 21 DFL game-seeds, 20 SkillCorner games, and the
6 in-sample Metrica units): largest |dP| and number of frames whose codes differ.
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
sys.path[:0] = [HERE, CAND, os.path.join(CAND, "mbr_decode")]
sys.path.append(os.path.join(CAND, "control_events"))
import stacker        # noqa: E402
import bench          # noqa: E402
import metrica_rules as mr   # noqa: E402

tsd = bench.tsd
WORK = os.environ["MTAH_WORK"]
PROBS = os.path.join(WORK, "probs")
GAMES = ["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"]
SEEDS = (2027, 2028)
MET_DIRS = ["adapt32/o_ls05_ref__tta4_all_a50", "met_ft/parent_m5c_skcft", "met_ft/parent_radar_pff",
            "met_ft/parent_p20M_pod"]
BENCH_MEMBERS = ["adapt32/o_ls05__tta4_all_a50", "m5c_C192ls_skcft", "radar_pff", "r5_ptr_ballc_p20M"]
PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, objective="multiclass", random_state=0,
              deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1)


def strict_labels(g):
    folder = os.path.join(bench.SAMPLE_DIR, g)
    game = tsd.load_epts_game(folder) if g == "Sample_Game_3" else tsd.load_csv_game(folder)
    prac, blocks = tsd.practice_labels(game)
    return mr.strict_labels(game, prac, blocks)


def met_unit(g, seed):
    blocks = bench.prepared(g, seed)[4]
    P = [np.load(os.path.join(PROBS, d, "met_{}_{}.npy".format(g, seed))).astype(np.float64) for d in MET_DIRS]
    return P, blocks


def train_rows(games):
    X, y = [], []
    for g in games:
        strict = strict_labels(g)
        for seed in SEEDS:
            P, blocks = met_unit(g, seed)
            runs = stacker.runs_of(blocks)
            n = len(P[0])
            assert n == len(strict)
            r = np.where(stacker.inside(n, runs) & (np.arange(n) % 5 == 0) & (strict < 3))[0]
            lab = strict[r].astype(np.int64)
            X += [stacker.feats(P, runs, r), stacker.feats(P, runs, r, swap=True)]
            y += [lab, np.where(lab == 0, 1, np.where(lab == 1, 0, lab))]
    return np.concatenate(X), np.concatenate(y)


def train(out):
    import lightgbm as lgb
    X, y = train_rows(GAMES)
    m = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    path = os.path.join(out, "stacker_s2.txt")
    m.booster_.save_model(path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    imp = m.booster_.feature_importance("gain").reshape(4, len(stacker.OFFS), 3)
    meta = dict(lgb_version=lgb.__version__, sha256=sha, params=PARAMS, rows=int(len(y)),
                class_counts=np.bincount(y).tolist(), offsets_frames=stacker.OFFS.tolist(), fps=25,
                features="member-major (TCN, MS-TCN, radar, pointer), then offset, then class (H, A, D); 84",
                member_dirs_metrica=MET_DIRS, games=GAMES, bench_seeds=list(SEEDS),
                gain_share_member=np.round(imp.sum((1, 2)) / imp.sum(), 3).tolist(),
                gain_share_offset=np.round(imp.sum((0, 2)) / imp.sum(), 3).tolist())
    with open(os.path.join(out, "stacker_s2_meta.json"), "w") as h:
        json.dump(meta, h, indent=1)
        h.write("\n")
    print("trained on {} rows (class counts {}); wrote {} sha256 {}".format(len(y), meta["class_counts"], path, sha))
    print("gain share per member {} per offset {}".format(meta["gain_share_member"], meta["gain_share_offset"]))


def check(model_file):
    import lightgbm as lgb
    sys.path.insert(0, os.path.join(CAND, "nn_pointer"))
    import finals                        # patches mbr_decode.data._path for adapt32/* and stack_aug/* names
    data = finals.data
    b = lgb.Booster(model_file=model_file)
    worst, ndiff, nunits = 0.0, 0, 0
    units = [("dfl", u) for u in data.units("dfl")] + [("skc", u) for u in data.units("skc")]
    for d, u in units:
        blocks, _ = data.info(d, u)
        P = [data.member(d, m, u)[0] for m in BENCH_MEMBERS]
        pr, c = stacker.predict(P, blocks, b)
        sp, sc = data.member(d, "stack_aug/S2_full", u)
        dp, dc = float(np.abs(pr.astype(np.float64) - sp).max()), int((c != sc).sum())
        worst, ndiff, nunits = max(worst, dp), ndiff + dc, nunits + 1
        print("{} {}: max|dP| {:.2e}, codes differ {}".format(d, u, dp, dc), flush=True)
    for g in GAMES:
        for seed in SEEDS:
            P, blocks = met_unit(g, seed)
            pr, c = stacker.predict(P, blocks, b)
            f = os.path.join(PROBS, "stack_aug", "S2_full", "insample_met_{}_{}".format(g, seed))
            sp, sc = np.load(f + ".npy").astype(np.float64), np.load(f + "_codes.npy")
            dp, dc = float(np.abs(pr.astype(np.float64) - sp).max()), int((c != sc).sum())
            worst, ndiff, nunits = max(worst, dp), ndiff + dc, nunits + 1
            print("met {} {}: max|dP| {:.2e}, codes differ {}".format(g, seed, dp, dc), flush=True)
    print("CHECK {}: {} units, max|dP| {:.2e}, frames with different codes {}".format(model_file, nunits, worst, ndiff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "check"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--model", default=stacker.MODEL)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.out)
    else:
        check(a.model)


if __name__ == "__main__":
    main()
