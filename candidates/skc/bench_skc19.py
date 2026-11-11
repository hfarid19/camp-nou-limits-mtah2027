#!/usr/bin/env python3
"""SkillCorner 2019/20 open data bench ("SKC19"): 9 top-league broadcast-tracked matches, never trained on.

    python candidates/skc/bench_skc19.py prepare
    python candidates/skc/bench_skc19.py run <pipeline.py> --name NAME
    python candidates/skc/bench_skc19.py score NAME [NAME ...] [--ref NAME]

Source: github.com/SkillCorner/opendata at commit 1b99d401 (MIT), in $MTAH_WORK/skc19/raw. Rule for its use:
docs/prereg/round5_prereg.txt, addendum 2026-11-10 12:45.

Inputs are the broadcast tracking AS GIVEN (no artificial degradation): every SkillCorner track_id becomes a tracklet in
one of 21 anonymous slots per team (as the match's tracking: fragmented, only on-screen players), 10 -> 25 fps by linear
interpolation inside a tracklet, NaN elsewhere; the ball is the detected ball, NaN when not detected.
Labels: SkillCorner's automatic per-frame possession group (home / away). Frames with no group are X (not graded): "none"
mixes dead ball, loose ball and broadcast cuts. The flag is "sticky" (it persists through passes and loose balls), so this
is a coarse, machine-made reference: a benchmark to report, not a selection set.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [HERE, CAND, ROOT, os.path.join(ROOT, "evaluation")]
WORK = os.environ["MTAH_WORK"]
RAW = os.path.join(WORK, "skc19", "raw")
CACHE = os.path.join(WORK, "skc19", "bench_cache")
PROBS = os.path.join(WORK, "probs", "skc19")
SLOTS_PER_TEAM = 21
RELEASE = 5          # source frames a track may be missing before its slot is freed
PITCH = (105.0, 68.0)

import possession_metrics as pm   # noqa: E402


def all_games():
    return sorted(str(m["id"]) for m in json.load(open(os.path.join(RAW, "matches.json"))))


def _interp(src, t):
    """src (n, ..., 2) at 10 fps with NaN, t fractional source index (m,) -> linear interpolation; NaN if either end is."""
    lo = np.clip(np.floor(t).astype(int), 0, len(src) - 1)
    hi = np.clip(lo + 1, 0, len(src) - 1)
    w = (t - lo).reshape((-1,) + (1,) * (src.ndim - 1))
    return src[lo] * (1 - w) + src[hi] * w


def prepared(game):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "skc19_{}.npz".format(game))
    if not os.path.isfile(path):
        md = json.load(open(os.path.join(RAW, game, "match_data.json")))
        frames = json.load(open(os.path.join(RAW, game, "structured_data.json")))
        home_id = md["home_team"]["id"]
        team_of = {p["trackable_object"]: (0 if p["team_id"] == home_id else 1) for p in md["players"]}
        ball_obj = md["ball"]["trackable_object"]
        ref_objs = {r["trackable_object"] for r in md.get("referees", [])}
        L, W = float(md["pitch_length"]), float(md["pitch_width"])
        grp_team = {"home team": 0, "home goalkeeper": 0, "away team": 1, "away goalkeeper": 1}
        slots_all, ball_all, gt_all, half_all = [], [], [], []
        for per in (1, 2):
            fr = [f for f in frames if f.get("period") == per]
            if not fr:
                continue
            f0, f1 = fr[0]["frame"], fr[-1]["frame"]
            n = f1 - f0 + 1
            S = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan)
            B = np.full((n, 2), np.nan)
            G = np.full(n, pm.X, np.int8)
            slot_of, last_seen = {}, {}          # track_id -> slot, slot -> last frame index
            for f in fr:
                i = f["frame"] - f0
                g = (f.get("possession") or {}).get("group")
                if g == "home team":
                    G[i] = pm.H
                elif g == "away team":
                    G[i] = pm.A
                for o in f["data"]:
                    to = o.get("trackable_object")
                    if to == ball_obj or o.get("group_name") == "ball":
                        B[i] = (o["x"], o["y"])
                        continue
                    if to in ref_objs or o.get("group_name") == "referee":
                        continue
                    team = team_of.get(to) if to is not None else grp_team.get(o.get("group_name"))
                    if team is None:
                        continue
                    key = (team, o["track_id"])
                    s = slot_of.get(key)
                    if s is None or last_seen.get(s, (None, -10 ** 9))[0] != key or i - last_seen[s][1] > RELEASE:
                        base = team * SLOTS_PER_TEAM
                        free = [k for k in range(base, base + SLOTS_PER_TEAM)
                                if k not in last_seen or i - last_seen[k][1] > RELEASE]
                        if not free:
                            continue
                        s = free[0]
                        slot_of[key] = s
                    S[i, s] = (o["x"], o["y"])
                    last_seen[s] = (key, i)
            # centre origin metres -> corner origin, scaled to 105 x 68
            S[..., 0] = (S[..., 0] + L / 2) * PITCH[0] / L
            S[..., 1] = (S[..., 1] + W / 2) * PITCH[1] / W
            B[:, 0] = (B[:, 0] + L / 2) * PITCH[0] / L
            B[:, 1] = (B[:, 1] + W / 2) * PITCH[1] / W
            t = np.arange(0, n - 1 + 1e-9, 0.4)          # 25 fps sample k at source index 0.4 k
            slots_all.append(_interp(S, t))
            ball_all.append(_interp(B, t))
            gt_all.append(G[np.clip(np.rint(t).astype(int), 0, n - 1)])
            half_all.append(np.full(len(t), per, np.int8))
        slots = np.concatenate(slots_all).astype(np.float32)
        ball = np.concatenate(ball_all).astype(np.float32)
        gt = np.concatenate(gt_all)
        half = np.concatenate(half_all)
        blocks = []
        for h in (1, 2):
            idx = np.where(half == h)[0]
            if len(idx):
                blocks.append((len(blocks) + 1, int(idx[0]), int(idx[-1])))
        slot_team = np.repeat(np.array([0, 1], np.int8), SLOTS_PER_TEAM)
        s0 = blocks[0][1]
        home_x = np.nanmean(slots[s0:s0 + 500, :SLOTS_PER_TEAM, 0])
        sign = 1.0 if home_x < PITCH[0] / 2 else -1.0
        tmp = path[:-4] + ".tmp{}.npz".format(os.getpid())
        np.savez(tmp, slots=slots, slot_team=slot_team, ball=ball, gt=gt, blocks=np.array(blocks), sign=sign)
        os.replace(tmp, path)
    z = np.load(path)
    return (z["slots"], z["slot_team"], z["ball"], z["gt"],
            [tuple(int(v) for v in b) for b in z["blocks"]], float(z["sign"]))


def load_candidate(path, name):
    spec = importlib.util.spec_from_file_location("skc19cand_" + name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(module)
    return module


def run(path, name, games):
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS"):
        if k in os.environ:
            raise SystemExit("{} is set: this bench uses full models only".format(k))
    out = os.path.join(PROBS, name)
    os.makedirs(out, exist_ok=True)
    todo = [g for g in games if not os.path.isfile(os.path.join(out, "skc19_{}_codes.npy".format(g)))]
    print("{} -> {}: {} games to run".format(path, out, len(todo)), flush=True)
    if not todo:
        return
    module = load_candidate(path, name)
    for g in todo:
        slots, slot_team, ball, gt, blocks, sign = prepared(g)
        t0 = time.time()
        pred = np.asarray(module.predict(slots.astype(np.float64), slot_team, ball.astype(np.float64), sign, blocks, [],
                                         ("Home", "Away"), PITCH, 25), np.int8)
        assert pred.shape == gt.shape
        np.save(os.path.join(out, "skc19_{}_codes.npy".format(g)), pred)
        print("  {} {}".format(g, fmt(score_codes(pred, g))), "{:.0f}s".format(time.time() - t0), flush=True)


def score_codes(pred, game):
    import bench
    gt = prepared(game)[3]
    r = {}
    for k, T in (("raw", None), ("T0.5", 0.5), ("T1", 1.0)):
        r[k] = pm.evaluate(gt if T is None else bench.controlled_labels(gt, T), pred, bootstrap_n=0)["spell_f1_mean"]
    r["fine3"] = float(np.mean([r["raw"], r["T0.5"], r["T1"]]))
    ev = pm.evaluate(gt, pred, bootstrap_n=0)
    r["n_pred"], r["n_gt"] = ev["n_pred_spells"], ev["n_gt_spells"]
    lab = (gt == pm.H) | (gt == pm.A)
    r["team_right"] = float((pred[lab] == gt[lab]).mean())
    r["team_wrong"] = float(((pred[lab] != gt[lab]) & ((pred[lab] == pm.H) | (pred[lab] == pm.A))).mean())
    r["dead"] = float((pred[lab] == pm.D).mean())
    r["labelled"] = float(lab.mean())
    return r


def fmt(r):
    return ("fine3 {fine3:.4f} raw {raw:.4f} | team right {team_right:.3f} wrong {team_wrong:.3f} dead {dead:.3f} | "
            "poss pred/ref {n_pred}/{n_gt}").format(**r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "score"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--name")
    ap.add_argument("--ref")
    ap.add_argument("--games", nargs="*")
    a = ap.parse_args()
    games = a.games or all_games()
    if a.cmd == "prepare":
        for g in games:
            slots, _, ball, gt, blocks, sign = prepared(g)
            lab = (gt == pm.H) | (gt == pm.A)
            print("{} rows {} labelled {:.3f} ball seen {:.3f} players on screen {:.1f} sign {:+.0f}".format(
                g, len(gt), lab.mean(), (~np.isnan(ball[:, 0])).mean(),
                (~np.isnan(slots[..., 0])).sum(1).mean(), sign), flush=True)
    elif a.cmd == "run":
        run(a.args[0], a.name, games)
    else:
        res = {}
        for name in a.args:
            res[name] = [score_codes(np.load(os.path.join(PROBS, name, "skc19_{}_codes.npy".format(g))), g) for g in games]
            m = {k: float(np.mean([r[k] for r in res[name]])) for k in res[name][0]}
            print("{:24s} n {} {}".format(name, len(games), fmt({k: (int(v) if k.startswith("n_") else v) for k, v in m.items()})))
        if a.ref:
            for name in a.args:
                if name == a.ref:
                    continue
                for key in ("fine3", "raw", "team_right", "team_wrong"):
                    d = np.array([x[key] - y[key] for x, y in zip(res[name], res[a.ref])])
                    print("paired {} vs {}: {} {:+.4f} ± {:.4f} (wins {}/{})".format(
                        name, a.ref, key, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))


if __name__ == "__main__":
    main()
