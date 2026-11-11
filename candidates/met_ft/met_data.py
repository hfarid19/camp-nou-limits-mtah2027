#!/usr/bin/env python3
"""MET-FT training data (round 5 prereg addendum 2026-11-10 11:50 B): Metrica's open sample games in the nn_pff /
pointer dataset format, with Metrica-rule controlled-possession labels.

    python -I candidates/met_ft/met_data.py [--out $MTAH_WORK/ds_met] [--seeds 41 42 43] [--verify]

<out>/ds/MET-G<k>_<seed>.npz   X (features.compute), y, yc, idx (one stride-5 phase: idx % 5 == seed % 5), names, kinds
<out>/raw/MET-G<k>_<seed>.npz  degraded slots / slot_team / ball, codes (strict labels), practice codes, blocks, sign
<out>/ptrb/...                 ball-centred pointer inputs (nn_pointer/prep_ptr.build_ballc)
<out>/owner/...                possessor slot target (DFL proxy rule of nn_pointer/make_owner.dfl_pcol on the clean
                               Metrica tracking, mapped to the degraded slot <= 3 m as make_owner.build)

Inputs are built exactly as bench.py builds the Metrica bench (tune_on_sample_data loaders, tsd.degrade with
tsd.SEED = seed and default_rng(seed), graded mask = practice labels != X, tsd.attack_sign, no ball offset), but with
training seeds (never the bench seeds 2027 / 2028). --verify rebuilds Sample Game 1 at bench seed 2027 in memory and
checks that the degraded inputs equal bench_cache's (nothing is written for it).
Labels: control_events/metrica_rules.strict_labels over tsd.practice_labels (H 0, A 1, D 2, X 3).
Reads only Metrica's open sample data.
"""

import argparse
import json
import multiprocessing.pool    # noqa: F401  (stdlib queue before nn_pff/queue.py is on sys.path)
import multiprocessing.queues  # noqa: F401
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), CAND, os.path.join(CAND, "nn_pff"),
                os.path.join(CAND, "control_events"), os.path.join(CAND, "nn_pointer"),
                os.path.join(CAND, "ball_radar_pff")]
import possession_metrics as pm    # noqa: E402
import tune_on_sample_data as tsd  # noqa: E402
import features as fe              # noqa: E402  (nn_pff/features.py, = the members' copies)
import data as D                   # noqa: E402  (nn_pff/data.py: change_target)
import metrica_rules as mr         # noqa: E402
import prep_ptr                    # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SAMPLE = os.path.join(WORK, "sample", "sample-data-master", "data")
GAMES = {"MET-G1": "Sample_Game_1", "MET-G2": "Sample_Game_2", "MET-G3": "Sample_Game_3"}
STRIDE = 5
CTRL_R, SLOT_R = 1.0, 3.0


def load_game(sample_name):
    folder = os.path.join(SAMPLE, sample_name)
    return tsd.load_epts_game(folder) if sample_name == "Sample_Game_3" else tsd.load_csv_game(folder)


def labels(game):
    prac, blocks = tsd.practice_labels(game)
    strict = mr.strict_labels(game, prac, blocks)
    return prac, strict, blocks


def degrade(game, prac, seed):
    """= bench.prepared's degradation with this seed."""
    tsd.SEED = seed
    slots, slot_team, ball = tsd.degrade(game, prac != pm.X, np.random.default_rng(seed))
    return slots, slot_team, ball


def possessor_cols(game, codes):
    """make_owner.dfl_pcol on the clean Metrica tracking with the label codes: latest controller (nearest player
    within 1 m of the ball) of the label's team in the current spell, -1 until the first one."""
    P, B, tp = game.players, game.ball, game.team_of_player
    d = np.linalg.norm(P - B[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    ctrl = np.where(d[np.arange(len(j)), j] < CTRL_R, j, -1)
    pcol = np.full(len(codes), -1, np.int16)
    cur, prev = -1, -9
    for t in range(len(codes)):
        c = codes[t]
        if c != prev:
            cur = -1
            prev = c
        if c > 1:
            continue
        if ctrl[t] >= 0 and tp[ctrl[t]] == c:
            cur = ctrl[t]
        pcol[t] = cur
    return pcol


def owner(game, pcol, idx, slots, st, codes):
    """make_owner.build: degraded slot nearest the clean possessor (same team, <= 3 m); -1 unknown, -2 dead."""
    P, tp = game.players, game.team_of_player
    own = np.full(len(idx), -1, np.int16)
    c = codes[idx]
    own[c == 2] = -2
    pc = pcol[idx]
    k = np.where((pc >= 0) & (c < 2))[0]
    r = idx[k]
    pos = P[r, pc[k]]
    d = np.linalg.norm(slots[r] - pos[:, None, :], axis=2)
    d = np.where(st[None, :] == tp[pc[k]][:, None], d, np.inf)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    ok = d[np.arange(len(k)), j] <= SLOT_R
    own[k[ok]] = j[ok]
    return own


def build(out, name, seed, game, prac, strict, blocks, pcol):
    ds_path = os.path.join(out, "ds", "{}_{}.npz".format(name, seed))
    raw_path = os.path.join(out, "raw", "{}_{}.npz".format(name, seed))
    own_path = os.path.join(out, "owner", "{}_{}.npz".format(name, seed))
    ptr_path = os.path.join(out, "ptrb", "{}_{}.npz".format(name, seed))
    if not (os.path.isfile(ds_path) and os.path.isfile(raw_path)):
        slots, slot_team, ball = degrade(game, prac, seed)
        sign = tsd.attack_sign(game, blocks)
        fs = fe.frame_sign(len(ball), sign, blocks)
        F, names, kinds = fe.compute(slots, slot_team, ball, fs, tsd.PITCH, tsd.FPS)
        graded = np.zeros(len(strict), bool)
        for _, a, b in blocks:
            graded[a:b + 1] = True
        graded &= strict != pm.X
        idx = np.where(graded)[0]
        idx = idx[(idx % STRIDE) == (seed % STRIDE)]
        tmp = raw_path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, slots=slots.astype(np.float32), slot_team=slot_team, ball=ball, codes=strict,
                            practice=prac, blocks=np.array(blocks), sign=sign)
        os.replace(tmp, raw_path)
        tmp = ds_path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, X=F[idx], y=strict[idx], yc=D.change_target(strict)[idx], idx=idx,
                            names=np.array(names), kinds=np.array(kinds))
        os.replace(tmp, ds_path)
    if not os.path.isfile(own_path):
        raw = np.load(raw_path)
        idx = np.load(ds_path)["idx"].astype(np.int64)
        own = owner(game, pcol, idx, raw["slots"], raw["slot_team"], raw["codes"])
        np.savez_compressed(own_path, own=own, idx=idx)
    msg = prep_ptr.build_ballc(ds_path, ptr_path)
    z = np.load(ds_path)
    o = np.load(own_path)["own"]
    alive = z["y"] < 2
    return "{} seed {}: rows {} ball missing (graded) {:.3f} owner known {:.2f} | {}".format(
        name, seed, len(z["idx"]), float(np.isnan(np.load(raw_path)["ball"][z["idx"], 0]).mean()),
        float((o[alive] >= 0).mean()), msg)


def align_report(name, game, prac, strict, blocks):
    """Events vs tracking: nearest player of the event's team to the (clean) ball at on-ball event start frames."""
    P, B, tp = game.players, game.ball, game.team_of_player
    dd, nearest_own, n = [], 0, 0
    for e in game.events:
        if e["type"] not in tsd.ON_BALL or e["team"] is None or not (0 <= e["start"] < len(B)):
            continue
        b = B[e["start"]]
        if np.isnan(b).any():
            continue
        d = np.linalg.norm(P[e["start"]] - b, axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        dt = d[tp == e["team"]].min()
        dd.append(dt)
        nearest_own += int(tp[int(np.argmin(d))] == e["team"])
        n += 1
    dd = np.array(dd)
    runs = [(a, b + 1) for _, a, b in blocks]
    graded = sum(b - a for a, b in runs)
    sp = len(pm.possession_spells(prac, runs))
    ss = len(pm.possession_spells(strict, runs))
    agree = float(np.mean(prac[np.concatenate([np.arange(a, b) for a, b in runs])] ==
                          strict[np.concatenate([np.arange(a, b) for a, b in runs])]))
    return ("{} ({}): on-ball events with ball tracked {}: event-team player within 1.5 m {:.3f}, 2 m {:.3f}, "
            "median {:.2f} m; nearest player is the event team's {:.3f} | possessions/90 practice {:.0f} strict {:.0f} "
            "| frame agreement practice vs strict {:.4f}").format(
        name, GAMES[name], n, float((dd <= 1.5).mean()), float((dd <= 2.0).mean()), float(np.median(dd)),
        nearest_own / max(n, 1), sp / graded * 90 * 60 * 25, ss / graded * 90 * 60 * 25, agree)


def verify():
    """bench seed 2027 rebuilt here == bench_cache (inputs the members see at Metrica bench time)."""
    sys.path.insert(0, CAND)
    import bench
    game = load_game("Sample_Game_1")
    prac, strict, blocks = labels(game)
    slots, slot_team, ball = degrade(game, prac, 2027)
    s2, st2, b2, gt2, bl2, sg2 = bench.prepared("Sample_Game_1", 2027)
    ok = (np.array_equal(slots, s2, equal_nan=True) and np.array_equal(ball, b2, equal_nan=True)
          and np.array_equal(slot_team, st2) and np.array_equal(prac, gt2) and list(blocks) == list(bl2)
          and float(tsd.attack_sign(game, blocks)) == float(sg2))
    print("verify Sample_Game_1 seed 2027 == bench_cache:", ok)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "ds_met"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--games", nargs="+", default=sorted(GAMES))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    assert not set(a.seeds) & {2027, 2028, 2029}, "bench seeds are never used for training"
    if a.verify:
        assert verify()
    for sub in ("ds", "raw", "owner", "ptrb"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    info = {}
    for name in a.games:
        game = load_game(GAMES[name])
        prac, strict, blocks = labels(game)
        print(align_report(name, game, prac, strict, blocks), flush=True)
        pcol = possessor_cols(game, strict)
        for seed in a.seeds:
            print(build(a.out, name, seed, game, prac, strict, blocks, pcol), flush=True)
        info[name] = dict(sample=GAMES[name], frames=len(prac), blocks=[list(map(int, b)) for b in blocks])
    with open(os.path.join(a.out, "config.json"), "w") as h:
        json.dump(dict(seeds=a.seeds, games=info, labels="metrica_rules.strict_labels(tsd.practice_labels)",
                       degrade="tune_on_sample_data.degrade (bench), tsd.SEED = seed"), h, indent=1)


if __name__ == "__main__":
    main()
