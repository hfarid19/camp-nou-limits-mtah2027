#!/usr/bin/env python3
"""Training data for nn_pff: the 48 PFF selection games + the 7 DFL games, degraded like the match.

    python -I candidates/nn_pff/make_dataset.py [--out $MTAH_WORK/ds_pff] [--jobs 2] [--only PFF|DFL]

Format = candidates/make_dataset_v2.py (so the TCN / MS-TCN recipes read it):
  <out>/ds/<name>_<seed>.npz   X (LDE features.compute, one stride-5 phase: idx % 5 == seed % 5),
                               y (0 H, 1 A, 2 D), yc (hand-over target), idx, names, kinds
  <out>/raw/<name>_<seed>.npz  degraded slots / slot_team / ball, full-resolution codes, blocks, sign
  <out>/raw/<name>_<seed>.json the degradation knobs drawn

Degradations per game (training seeds; the bench seeds 2027-2029 are never used here). Default
(ds_pff, used by the nn_pff architecture experiments):
  41, 42  degrade_v2 with make_dataset_v2.draw_params (randomised around the fitted defaults)
  44      degrade_v4 (dead-ball islands, rolling exits, long open-play gaps: calibrated to the
          match's ball at stoppages) with knobs randomised around its defaults (draw_params_v4)
  every one + make_dataset_v2's AR(1) ball offset (sd drawn up to 0.5 m).
DFL: seeds 41-43 already exist as $MTAH_WORK/ds_v2s (degrade_v2 randomised, sign fixed) and are
hard-linked; only seed 44 (degrade_v4) is built here.

Labels: PFF = the chosen "touch" reading ($MTAH_WORK/pff_wc2022/npz, candidates/pff/labels.py);
DFL = the raw BallStatus / BallPossession flags. Attack direction: the bench's kick-off rule
(tune_on_sample_data.attack_sign), never learned_dfl_evidence.make_dataset.attack_sign (which
reverses DFL J03WR9 and 6 PFF games); check_signs.py verifies it against the goalkeepers.
The 16 holdout games (arch/nn_pff/holdout_games.json) are refused.

ds_pff2 (the "v5c+PFF" retrain, inputs as in ds_v2s: degrade_v2 only):
    --out $MTAH_WORK/ds_pff2 --pff-seeds 41 42 43 --dfl-new-seeds --link-from $MTAH_WORK/ds_pff
(seeds 41 / 42 are bit-identical to ds_pff's and hard-linked; DFL = ds_v2s seeds 41-43 only).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
LDE = os.path.join(CAND, "learned_dfl_evidence")
sys.path[:0] = [LDE, CAND, ROOT]
import features as fe              # noqa: E402  learned_dfl_evidence/features.py (= the members' copies)
import make_dataset as lde_ds      # noqa: E402
import make_dataset_v2 as mdv2     # noqa: E402
import degrade_v2 as dv2           # noqa: E402
import degrade_v4 as dv4           # noqa: E402
import tune_on_sample_data as tsd  # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
STRIDE, FPS, PITCH = 5, 25, (105.0, 68.0)
PFF_SEEDS = (41, 42, 44)
DFL_LINK_SEEDS = (41, 42, 43)
DFL_NEW_SEEDS = (44,)
V4_SEEDS = {44}


def holdout():
    return set(json.load(open(os.path.join(WORK, "arch", "nn_pff", "holdout_games.json")))["games"])


def selection_games():
    f = json.load(open(os.path.join(WORK, "arch", "nn_pff", "selection_folds.json")))["folds"]
    g = sorted(x for fold in f for x in fold)
    assert not set(g) & holdout() and len(g) == 48
    return g


def draw_params_v4(rng):
    """degrade_v4 knobs randomised around its DEFAULTS (player knobs as make_dataset_v2)."""
    return dict(roll_s=rng.uniform(0.3, 1.0), p_roll=rng.uniform(0.2, 0.6), reveal_s=rng.uniform(1.0, 4.0),
                park_s=rng.uniform(0.2, 1.2), isl_per_min=rng.uniform(1.0, 3.0), isl_s=rng.uniform(2.0, 5.0),
                isl_offset_m=rng.uniform(0.8, 2.5), alive_gaps_per90=rng.uniform(8, 25),
                alive_gap_min_s=rng.uniform(3.0, 6.0), alive_gap_s=rng.uniform(1.5, 4.0),
                micro_per90=rng.uniform(5, 20), crowd_boost=rng.uniform(1.5, 4.0),
                tracklet_s=rng.uniform(40, 160), player_gap_s=rng.uniform(3, 12),
                slots_per_team=int(rng.integers(21, 25)), smooth_frames=int(rng.choice([7, 9, 11, 13, 15, 19])),
                ball_smooth_frames=int(rng.choice([1, 3, 5])), ball_offset_sd=rng.uniform(0.0, 0.5))


def degrade(g, name, seed):
    codes, blocks = lde_ds.labels_blocks(g)
    rng = np.random.default_rng([seed, lde_ds.game_key(name)])
    v4 = seed in V4_SEEDS
    p = draw_params_v4(rng) if v4 else mdv2.draw_params(rng)
    graded = codes != 3
    fn = dv4.degrade_v4 if v4 else dv2.degrade_v2
    slots, slot_team, ball = fn(g["players"].astype(np.float64), g["team_of_player"].astype(int),
                                g["ball"].astype(np.float64), g["status"], graded,
                                np.random.default_rng([seed, lde_ds.game_key(name), 1]), p)
    n = len(ball)
    a = np.exp(-1.0 / FPS)
    eps = rng.normal(0, 1, (n, 2)) * p["ball_offset_sd"] * np.sqrt(1 - a * a)
    off = np.zeros((n, 2))
    for t in range(1, n):
        off[t] = a * off[t - 1] + eps[t]
    ball = (ball + off).astype(np.float32)
    sign = tsd.attack_sign(tsd.Game(name, g["players"].astype(np.float64), g["team_of_player"].astype(int),
                                    g["ball"].astype(np.float64), []), blocks)
    p["degrade"] = "v4" if v4 else "v2"
    return slots, slot_team, ball, codes, blocks, sign, p


def job(args):
    path, name, seed, out = args
    ds_path = os.path.join(out, "ds", "{}_{}.npz".format(name, seed))
    raw_path = os.path.join(out, "raw", "{}_{}.npz".format(name, seed))
    if os.path.isfile(ds_path) and os.path.isfile(raw_path):
        return "{} seed {}: exists".format(name, seed)
    t0 = time.time()
    g = lde_ds.load_game(path)
    slots, slot_team, ball, codes, blocks, sign, p = degrade(g, name, seed)
    fs = fe.frame_sign(len(ball), sign, blocks)
    F, names, kinds = fe.compute(slots, slot_team, ball, fs, PITCH, FPS)
    graded = np.zeros(len(codes), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    graded &= codes != 3
    idx = np.where(graded)[0]
    idx = idx[(idx % STRIDE) == (seed % STRIDE)]
    tmp = ds_path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, X=F[idx], y=codes[idx], yc=lde_ds.change_target(codes)[idx], idx=idx,
                        names=np.array(names), kinds=np.array(kinds))
    os.replace(tmp, ds_path)
    tmp = raw_path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, slots=slots.astype(np.float32), slot_team=slot_team, ball=ball, codes=codes,
                        blocks=np.array(blocks), sign=sign)
    os.replace(tmp, raw_path)
    with open(raw_path[:-4] + ".json", "w") as handle:
        json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else
                       int(v) if isinstance(v, (np.integer,)) else v) for k, v in p.items()}, handle)
    return "{} seed {} ({}): {} rows, ball missing {:.2f}, sign {:+.0f}, {:.0f}s".format(
        name, seed, p["degrade"], len(idx), float(np.isnan(ball[graded, 0]).mean()), sign, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "ds_pff"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=None, choices=[None, "PFF", "DFL"])
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--pff-seeds", type=int, nargs="+", default=list(PFF_SEEDS))
    ap.add_argument("--dfl-new-seeds", type=int, nargs="*", default=list(DFL_NEW_SEEDS))
    ap.add_argument("--link-from", default=None, help="hard-link existing PFF files from this dataset")
    args = ap.parse_args()
    for sub in ("ds", "raw"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    hold = holdout()
    jobs = []
    if args.only in (None, "DFL"):
        dfl = sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "idsse", "npz")) if f.endswith(".npz"))
        for m in dfl:
            for s in DFL_LINK_SEEDS:
                for sub in ("ds", "raw"):
                    src = os.path.join(WORK, "ds_v2s", sub, "{}_{}.npz".format(m, s))
                    dst = os.path.join(args.out, sub, "{}_{}.npz".format(m, s))
                    if not os.path.exists(dst):
                        os.link(src, dst)
            for s in args.dfl_new_seeds:
                jobs.append((os.path.join(WORK, "idsse", "npz", m + ".npz"), m, s, args.out))
    if args.only in (None, "PFF"):
        for gname in (args.games or selection_games()):
            assert gname not in hold, "holdout game " + gname
            for s in args.pff_seeds:
                if args.link_from:
                    src = [os.path.join(args.link_from, sub, "PFF-{}_{}.npz".format(gname, s)) for sub in ("ds", "raw")]
                    if all(os.path.isfile(f) for f in src):
                        for f, sub in zip(src, ("ds", "raw")):
                            dst = os.path.join(args.out, sub, os.path.basename(f))
                            if not os.path.exists(dst):
                                os.link(f, dst)
                        j = src[1][:-4] + ".json"
                        if os.path.isfile(j) and not os.path.exists(os.path.join(args.out, "raw", os.path.basename(j))):
                            os.link(j, os.path.join(args.out, "raw", os.path.basename(j)))
                        continue
                jobs.append((os.path.join(WORK, "pff_wc2022", "npz", "PFF-" + gname + ".npz"), "PFF-" + gname, s, args.out))
    with open(os.path.join(args.out, "config.json"), "w") as handle:
        json.dump(dict(pff_seeds=args.pff_seeds, dfl_link_seeds=DFL_LINK_SEEDS, dfl_new_seeds=args.dfl_new_seeds,
                       v4_seeds=sorted(V4_SEEDS), stride=STRIDE), handle, indent=1)
    if args.jobs <= 1:
        for j in jobs:
            print(job(j), flush=True)
        return
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for msg in pool.imap_unordered(job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
