#!/usr/bin/env python3
"""Degrade the parsed DFL/IDSSE games (domain randomisation) and compute training features.

    python -I candidates/learned_dfl_evidence/make_dataset.py [--cache DIR]

Per game and per seed in SEEDS:
  1. tune_on_sample_data.degrade (the bench's own degradation: anonymous fragmented slots,
     crowd-dependent ball gaps) with target_missing ~ U(0.33, 0.48), mean_tracklet_s ~ U(30, 100)
  2. on top, here only: AR(1) ball offset (sd U(0, 1.3) m per axis, tau 1 s) + white noise
     (sd U(0.05, 0.2) m) on the seen ball; ghost duplicate slots on ~2% of frames (a player
     copied 0.5-1 m away into a free slot); occasional far-side player dropout; a low-passed
     ball on half of the degradations (round 1). Round 2 (defaults; round-1 dataset with
     --drop-p 0 --blackout-p 0): extra short 0.5-2 s ball dropouts on half of the degradations,
     and stoppage blackouts on half (ball hidden over whole BallStatus-0 stoppages, some live
     4 s windows, tracked-player collapse inside some stoppages; see _blackout). The settings
     are recorded in <cache>/ds/config.json and copied into model_meta.json["datasets"].
  3. features.compute, labels from the tracking flags (D where BallStatus 0, else H/A from
     BallPossession), stride-5 subsample of graded frames -> <cache>/ds/<match>_<seed>.npz
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import tune_on_sample_data as tsd   # noqa: E402
import features as fe               # noqa: E402

warnings.filterwarnings("ignore")
# raw data + caches: $IDSSE_CACHE, else <candidate>/external/idsse (git-ignored); --cache overrides
DEFAULT_CACHE = os.environ.get("IDSSE_CACHE", os.path.join(HERE, "external", "idsse"))
SEEDS = [11, 12, 13]
STRIDE = 5
PITCH = (105.0, 68.0)
FPS = 25


def game_key(match):
    """Stable per-match integer for the randomisation streams (independent of file order)."""
    return zlib.crc32(match.encode()) % 100000


CHANGE_HW = 4


def change_target(codes, hw=CHANGE_HW):
    """1 within +-hw frames of a direct H <-> A change of the tracking possession flag."""
    c = np.zeros(len(codes), bool)
    t = np.where((codes[1:] != codes[:-1]) & (codes[1:] < 2) & (codes[:-1] < 2))[0] + 1
    for k in range(-hw, hw + 1):
        c[np.clip(t + k, 0, len(c) - 1)] = True
    return c.astype(np.int8)


def load_game(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def labels_blocks(g):
    st, po, half = g["status"], g["poss"], g["half"]
    codes = np.full(len(st), 3, np.int8)
    codes[(st == 1) & (po == 1)] = 0
    codes[(st == 1) & (po == 2)] = 1
    codes[st == 0] = 2
    blocks = []
    for h in (1, 2):
        idx = np.where((half == h) & (codes != 3))[0]
        if len(idx):
            blocks.append((h, int(idx[0]), int(idx[-1])))
    return codes, blocks


def attack_sign(g, blocks):
    _, s, e = blocks[0]
    home = g["players"][s:e + 1, g["team_of_player"] == 0, 0]
    return 1.0 if np.nanmean(home) < PITCH[0] / 2 else -1.0


def _runs_mask(n, rate, mean_len, rng):
    m = np.zeros(n, bool)
    starts = np.where(rng.random(n) < rate)[0]
    lens = (rng.exponential(mean_len, len(starts)) + 1).astype(int)
    for s, l in zip(starts, lens):
        m[s:s + l] = True
    return m


SMOOTH_P = 0.5   # share of degradations with a low-passed ball (--smooth-p; 0 = round-0 dataset)
# Round 2 options, each on its own RNG stream so the round-1 draws are unchanged. The defaults
# below build the round-2 dataset of model_d / model_c; model_ha is still trained on the round-1
# dataset (--drop-p 0 --blackout-p 0). Wider offsets (--offset-sd-max 2 --tau-max-s 3) were
# tried and rejected: they cost the team model ~0.015 clean spellF1 on the bench.
OFFSET_SD_MAX = 1.3   # AR(1) ball offset sd ~ U(0, OFFSET_SD_MAX) m per axis
TAU_MAX_S = 1.0       # AR(1) time constant ~ U(1, TAU_MAX_S) s (1: fixed 1 s, as in round 1)
DROP_P = 0.5          # share of degradations with extra short (0.5-2 s) ball dropouts
BLACKOUT_P = 0.5      # share of degradations with stoppage blackouts (see _blackout)


def _nan_mean(values, hw):
    """Centred NaN-aware moving mean over 2*hw+1 frames."""
    v = np.where(np.isnan(values), 0.0, values).astype(np.float64)
    c = (~np.isnan(values)).astype(np.float64)
    cv, cc = np.concatenate(([0], np.cumsum(v))), np.concatenate(([0], np.cumsum(c)))
    i = np.arange(len(values))
    lo, hi = np.maximum(0, i - hw), np.minimum(len(values), i + hw + 1)
    k = cc[hi] - cc[lo]
    return np.where(k > 0, (cv[hi] - cv[lo]) / np.maximum(k, 1), np.nan)


def _blackout(codes, graded, brng):
    """Stoppage blackouts: the competition's missing ball sits mostly in long gaps (85% of its
    missing time, 41% of graded frames inside gaps > 3 s, a label-free statistic of the match
    input), which the bench degradation never produces on DFL (gap p99 ~5-7 s).
    -> (hide_ball mask, collapse mask): the ball hidden over whole BallStatus 0 stoppages with
    probability p_stop ~ U(0.5, 0.95) (edges jittered), plus live windows (exponential, mean 4 s)
    covering ~U(0.02, 0.08) of the alive frames; inside ~30% of the hidden stoppages the tracked
    player count collapses (slots dropped, see degrade_dfl)."""
    n = len(codes)
    hide = np.zeros(n, bool)
    collapse = np.zeros(n, bool)
    p_stop = brng.uniform(0.5, 0.95)
    dead = (codes == 2) & graded
    for s, e in zip(*fe.runs_of(dead)):
        if e - s < FPS or brng.random() >= p_stop:
            continue
        a = int(np.clip(s + brng.uniform(-0.5, 1.0) * FPS, 0, n))
        b = int(np.clip(e + brng.uniform(-1.0, 0.5) * FPS, a, n))
        hide[a:b] = True
        if brng.random() < 0.3:
            collapse[a:b] = True
    alive = graded & (codes < 2)
    share = brng.uniform(0.02, 0.08)
    mean_len = 4.0 * FPS
    rate = share / mean_len
    for t in np.flatnonzero(alive & (brng.random(n) < rate)):
        hide[t:t + int(brng.exponential(mean_len)) + FPS] = True
    return hide, collapse


def degrade_dfl(g, seed, game_index):
    """-> slots, slot_team, ball, codes, blocks, sign_half1 for one randomised degradation."""
    codes, blocks = labels_blocks(g)
    rng = np.random.default_rng([seed, game_index])
    target = float(rng.uniform(0.33, 0.48))
    tracklet = float(rng.uniform(30, 100))
    game = tsd.Game("dfl", g["players"].astype(np.float64), g["team_of_player"].astype(int),
                    g["ball"].astype(np.float64), [])
    tsd.SEED = seed * 100 + game_index
    graded = codes != 3
    brng = np.random.default_rng([seed, game_index, 5])
    blackout = brng.random() < BLACKOUT_P
    if blackout:      # the long gaps replace part of the short random ones
        target *= float(brng.uniform(0.25, 0.6))
    slots, slot_team, ball = tsd.degrade(game, graded, np.random.default_rng([seed, game_index, 1]),
                                         target_missing=target, mean_tracklet_s=tracklet)
    n = len(ball)
    # AR(1) offset + white noise on the seen ball
    sd = rng.uniform(0, 1.3)                           # same single draw as round 1
    if OFFSET_SD_MAX != 1.3:
        sd *= OFFSET_SD_MAX / 1.3
    tau_s = 1.0
    if TAU_MAX_S > 1.0:
        tau_s = float(np.random.default_rng([seed, game_index, 3]).uniform(1.0, TAU_MAX_S))
    a = np.exp(-1.0 / (tau_s * FPS))
    eps = rng.normal(0, 1, (n, 2)) * sd * np.sqrt(1 - a * a)
    off = np.zeros((n, 2))
    off[0] = rng.normal(0, sd, 2)
    for t in range(1, n):
        off[t] = a * off[t - 1] + eps[t]
    ball = ball + off + rng.normal(0, rng.uniform(0.05, 0.2), (n, 2))
    # low-pass the seen ball on half of the degradations (the competition ball is smooth: step
    # speed median ~5 m/s); separate stream, so every other draw above and below is unchanged
    srng = np.random.default_rng([seed, game_index, 2])
    smooth_hw = int(srng.integers(4, 7)) if srng.random() < SMOOTH_P else 0
    if smooth_hw:
        seen = ~np.isnan(ball[:, 0])
        ball = np.stack([_nan_mean(ball[:, i], smooth_hw) for i in (0, 1)], 1)
        ball[~seen] = np.nan
    # extra short ball dropouts (0.5-2 s) on DROP_P of the degradations, own stream
    drng = np.random.default_rng([seed, game_index, 4])
    if drng.random() < DROP_P:
        rate = drng.uniform(0.05, 0.2) / 30
        for t in np.flatnonzero(drng.random(n) < rate):
            ball[t:t + int(drng.integers(12, 50))] = np.nan
    ball = ball.astype(np.float32)
    if blackout:
        hide, collapse = _blackout(codes, graded, brng)
        ball[hide] = np.nan
    # ghost duplicates on ~2% of frames
    ghost = _runs_mask(n, 0.02 / 25, 25, rng)
    S = slots.shape[1]
    for s, e in zip(*fe.runs_of(ghost)):
        cand = np.where(~np.isnan(slots[s, :, 0]))[0]
        free = np.where(np.isnan(slots[s:e, :, 0]).all(0))[0]
        if not len(cand) or not len(free):
            continue
        j = int(rng.choice(cand))
        free = free[slot_team[free] == slot_team[j]]
        if not len(free):
            continue
        k = int(rng.choice(free))
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.5, 1.0)
        slots[s:e, k] = slots[s:e, j] + np.array([r * np.cos(ang), r * np.sin(ang)], np.float32)
    # far-side dropout: windows where players beyond 75% of the width (camera far side) vanish
    far_hi = rng.random() < 0.5
    drop = _runs_mask(n, rng.uniform(0.01, 0.05) / (5 * FPS), 5 * FPS, rng)
    y = slots[:, :, 1]
    far = (y > 0.75 * PITCH[1]) if far_hi else (y < 0.25 * PITCH[1])
    kill = far & drop[:, None]
    slots[kill] = np.nan
    if blackout:      # tracking collapse inside hidden stoppages: keep 5-15 tracked players
        for s, e in zip(*fe.runs_of(collapse)):
            keep = int(brng.integers(5, 16))
            vis = np.where((~np.isnan(slots[s:e, :, 0])).any(0))[0]
            if len(vis) > keep:
                drop_j = brng.choice(vis, len(vis) - keep, replace=False)
                slots[s:e, drop_j] = np.nan
    sign = attack_sign(g, blocks)
    return slots, slot_team, ball, codes, blocks, sign, dict(target=target, tracklet=tracklet,
                                                             ball_sd=sd, smooth_hw=smooth_hw,
                                                             tau_s=tau_s, blackout=blackout)


def main():
    global SMOOTH_P, OFFSET_SD_MAX, TAU_MAX_S, DROP_P, BLACKOUT_P
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--smooth-p", type=float, default=SMOOTH_P)
    ap.add_argument("--offset-sd-max", type=float, default=OFFSET_SD_MAX)
    ap.add_argument("--tau-max-s", type=float, default=TAU_MAX_S)
    ap.add_argument("--drop-p", type=float, default=DROP_P)
    ap.add_argument("--blackout-p", type=float, default=BLACKOUT_P)
    args = ap.parse_args()
    SMOOTH_P, OFFSET_SD_MAX, TAU_MAX_S = args.smooth_p, args.offset_sd_max, args.tau_max_s
    DROP_P, BLACKOUT_P = args.drop_p, args.blackout_p
    out_dir = os.path.join(args.cache, "ds")
    os.makedirs(out_dir, exist_ok=True)
    config = dict(seeds=SEEDS, stride=STRIDE, smooth_p=SMOOTH_P, offset_sd_max=OFFSET_SD_MAX,
                  tau_max_s=TAU_MAX_S, drop_p=DROP_P, blackout_p=BLACKOUT_P)
    cpath = os.path.join(out_dir, "config.json")
    if os.path.isfile(cpath):          # existing files are reused, so the settings must match
        with open(cpath) as handle:
            assert json.load(handle) == config, "{} was built with other settings".format(out_dir)
    with open(cpath, "w") as handle:
        json.dump(config, handle, indent=1)
    paths = sorted(glob.glob(os.path.join(args.cache, "npz", "DFL-MAT-*.npz")))
    for path in paths:
        match = os.path.basename(path)[:-4]
        gi = game_key(match)
        g = None
        for seed in SEEDS:
            out = os.path.join(out_dir, "{}_{}.npz".format(match, seed))
            if os.path.isfile(out):
                continue
            if g is None:
                g = load_game(path)
            t0 = time.time()
            slots, slot_team, ball, codes, blocks, sign, info = degrade_dfl(g, seed, gi)
            fs = fe.frame_sign(len(ball), sign, blocks)
            F, names, kinds = fe.compute(slots, slot_team, ball, fs, PITCH, FPS)
            graded = np.zeros(len(codes), bool)
            for _, a, b in blocks:
                graded[a:b + 1] = True
            graded &= codes != 3
            idx = np.where(graded)[0]
            idx = idx[(idx % STRIDE) == (seed % STRIDE)]
            yc = change_target(codes)
            np.savez_compressed(out, X=F[idx], y=codes[idx], yc=yc[idx], idx=idx,
                                names=np.array(names),
                                kinds=np.array(kinds))
            print("{} seed {}: {} rows, missing {:.2f}, tracklet {:.0f}s, ball sd {:.2f}, "
                  "smooth hw {}, tau {:.1f}s, blackout {:d}, {:.0f}s".format(match, seed, len(idx),
                                                np.isnan(ball[graded, 0]).mean(), info["tracklet"],
                                                info["ball_sd"], info["smooth_hw"], info["tau_s"],
                                                int(info["blackout"]), time.time() - t0),
                  flush=True)


if __name__ == "__main__":
    main()
