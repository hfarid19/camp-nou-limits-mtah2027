#!/usr/bin/env python3
"""Stacked-LightGBM training features: LDE features recomputed on the Ball Radar imputed ball,
plus radar evidence columns (features_br.compute), on domain-randomised DFL degradations.

    python -I candidates/ball_radar/feat_data.py --out <scratch>/arch/ball_radar/feat \
        --radar k [k ...]       # k = fold index (fold_radar_k.npz) or 'full' (radar_full.npz)

For radar model k (never saw game k; 'full' saw all games) features are written for EVERY game:
games j != k are training rows for the LightGBM fold k (radar in-sample there, exactly as the
full LightGBM sees in-sample radar on all games), game k itself is out-of-sample and is only
used to pick the boosting rounds (LOGO curve), never as training rows. So the LightGBM fold k
depends on no data of game k at all (strict leave-one-game-out).
Degradation: learned_dfl_evidence/make_dataset.degrade_dfl, seed SEED (not used by the radar
training seeds 31-33, the LDE datasets 11-13 or the bench 2027 / 21); rows: stride 5.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LDE = os.path.join(os.path.dirname(HERE), "learned_dfl_evidence")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, LDE)
sys.path.insert(0, HERE)
import make_dataset as md          # noqa: E402  (LDE's degradation)


def _kickoff_sign(g, blocks):
    """ball_radar_v3: the bench's attack-direction rule (tune_on_sample_data.attack_sign: home
    players over the first 50 frames of the first half). make_dataset.attack_sign averages over the
    whole first half, which reverses DFL-MAT-J03WR9 (home mean 54.9 m, but 45.7 m at the kick-off)."""
    _, s, _ = blocks[0]
    home = g["players"][s:s + 50, g["team_of_player"] == 0, 0]
    return 1.0 if np.nanmean(home) < md.PITCH[0] / 2 else -1.0


md.attack_sign = _kickoff_sign     # degrade_dfl looks it up at call time
import features_br as fb           # noqa: E402
import radar_common as rc          # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

SEED = 51
STRIDE = 5
SCR = WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--radar", nargs="+", required=True)
    ap.add_argument("--games", type=int, nargs="*", default=None)
    args = ap.parse_args()
    npz = os.path.join(SCR, "idsse", "npz")
    matches = sorted(f[:-4] for f in os.listdir(npz) if f.endswith(".npz"))
    os.makedirs(args.out, exist_ok=True)
    W = {}
    for k in args.radar:
        p = os.path.join(SCR, "arch", "ball_radar_v3",
                         "radar_full.npz" if k == "full" else "folds/fold_radar_{}.npz".format(k))
        W[k] = rc.load_weights(p)
    games = args.games if args.games is not None else range(len(matches))
    for j in games:
        todo = [k for k in args.radar
                if not os.path.isfile(os.path.join(args.out, "r{}_g{}.npz".format(k, j)))]
        if not todo:
            continue
        g = md.load_game(os.path.join(npz, matches[j] + ".npz"))
        slots, slot_team, ball, codes, blocks, sign, _ = md.degrade_dfl(g, SEED, md.game_key(matches[j]))
        fs = fb.frame_sign(len(ball), sign, blocks)
        graded = np.zeros(len(codes), bool)
        for _, a, b in blocks:
            graded[a:b + 1] = True
        graded &= codes != 3
        idx = np.where(graded)[0]
        idx = idx[(idx % STRIDE) == (SEED % STRIDE)]
        yc = md.change_target(codes)
        for k in todo:
            t0 = time.time()
            r = rc.run(W[k], slots, slot_team, ball, sign, blocks, md.FPS)
            F, names, kinds = fb.compute(slots, slot_team, ball, fs, md.PITCH, md.FPS,
                                         ball_imp=r["ball"], radar=r)
            clean = g["ball"].astype(np.float64)
            gap = np.isnan(ball[:, 0]) & graded & (codes < 2)
            err = np.sqrt(((r["ball"][gap] - clean[gap]) ** 2).sum(1)).mean()
            lin = rc.interp_ball(ball)
            lerr = np.sqrt(((lin[gap] - clean[gap]) ** 2).sum(1)).mean()
            np.savez_compressed(os.path.join(args.out, "r{}_g{}.npz".format(k, j)), X=F[idx],
                                y=codes[idx], yc=yc[idx], idx=idx, names=np.array(names),
                                kinds=np.array(kinds))
            print("radar {} game {} rows {} alive-gap err {:.2f} m (linear interp {:.2f} m) {:.0f}s".format(
                k, j, len(idx), err, lerr, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
