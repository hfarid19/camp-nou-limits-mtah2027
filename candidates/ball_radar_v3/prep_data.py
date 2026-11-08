#!/usr/bin/env python3
"""Build Ball Radar training sequences from the open DFL/IDSSE games (never Metrica, never the
competition match).

    python -I candidates/ball_radar/prep_data.py --out <scratch>/arch/ball_radar/seq --seeds 31 32

Per game and seed: learned_dfl_evidence/make_dataset.degrade_dfl (domain-randomised degradation
like the competition's automatic tracking: anonymous fragmented slots, 33-48% ball missing,
AR(1) ball offset, low-passed ball, stoppage blackouts, ghosts, far-side dropouts) with seeds
that neither the LightGBM datasets (11-13) nor the bench (2027, 21) use. Inputs are sampled
every STEP frames inside the graded halves (phase = seed % STEP); targets come from the CLEAN
DFL ball and the tracking flags:
    ball   clean ball, canonical / 10 m
    code   0 H, 1 A, 2 D (BallPossession / BallStatus), 3 ungraded
    owner  player-level possession (Ball Radar's intermediate target): the slot nearest the clean
           ball when it is within OWN_R m of a tracked slot ("controller"), else the next
           controller within the same alive spell; S (= dead class) while BallStatus is 0;
           -1 (ignored) when unknown.
--eval writes a bench-like degradation (tune_on_sample_data.degrade, the plain one) for the
ball-error diagnostic instead.
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
import features as fe              # noqa: E402
import tune_on_sample_data as tsd  # noqa: E402
import radar_common as rc          # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

OWN_R = 1.0


def owner_target(slots, ball_clean, codes):
    n, S = slots.shape[:2]
    d = np.sqrt(((slots - ball_clean[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    dm = d[np.arange(n), j]
    ctrl = (dm < OWN_R) & (codes < 2)
    own = np.full(n, -1, np.int64)
    own[ctrl] = j[ctrl]
    # next controller, backwards fill inside alive spells
    nxt = -1
    for t in range(n - 1, -1, -1):
        if codes[t] >= 2:
            nxt = -1
            continue
        if ctrl[t]:
            nxt = own[t]
        else:
            own[t] = nxt
    own[codes == 2] = S
    return own


def build(g, seed, gi, eval_mode=False):
    if eval_mode:
        codes, blocks = md.labels_blocks(g)
        game = tsd.Game("dfl", g["players"].astype(np.float64), g["team_of_player"].astype(int),
                        g["ball"].astype(np.float64), [])
        tsd.SEED = seed
        slots, slot_team, ball = tsd.degrade(game, codes != 3, np.random.default_rng(seed))
        sign = md.attack_sign(g, blocks)
    else:
        slots, slot_team, ball, codes, blocks, sign, _ = md.degrade_dfl(g, seed, gi)
    sf = fe.frame_sign(len(ball), sign, blocks)
    clean = g["ball"].astype(np.float64)
    own = owner_target(slots, clean, codes)
    out = {}
    for h, first, last in blocks:
        frames = np.arange(first + seed % rc.STEP, last + 1, rc.STEP)
        inp = rc.prepare(slots, slot_team, ball, sf, md.FPS, frames)
        for k, v in inp.items():
            out["h{}_{}".format(h, k)] = v
        out["h{}_ball".format(h)] = rc.canon(clean[frames], sf[frames]).astype(np.float32)
        out["h{}_code".format(h)] = codes[frames].astype(np.int8)
        out["h{}_owner".format(h)] = own[frames].astype(np.int16)
        out["h{}_seen".format(h)] = (~np.isnan(ball[frames, 0])).astype(np.int8)
        out["h{}_frames".format(h)] = frames.astype(np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=WORK + "/idsse/npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[31, 32])
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    matches = sorted(f[:-4] for f in os.listdir(args.npz) if f.endswith(".npz"))
    for gi, match in enumerate(matches):
        g = None
        for seed in args.seeds:
            path = os.path.join(args.out, "{}_{}{}.npz".format(match, "e" if args.eval else "", seed))
            if os.path.isfile(path):
                continue
            if g is None:
                g = md.load_game(os.path.join(args.npz, match + ".npz"))
            t0 = time.time()
            out = build(g, seed, md.game_key(match), args.eval)
            np.savez(path, **out)
            print(match, seed, "{:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
