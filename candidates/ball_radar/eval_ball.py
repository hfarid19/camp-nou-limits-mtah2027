#!/usr/bin/env python3
"""Ball-position error of the Ball Radar fold models on held-out DFL games (diagnostic).

    python -I candidates/ball_radar/eval_ball.py --seq <scratch>/arch/ball_radar/seq \
        --folds <scratch>/arch/ball_radar/folds [--tag radar] [--games 0 1 ...]

For each game gi, fold model fold_<tag>_<gi>.npz (never saw game gi) runs on the bench-like
degradation of that game (tune_on_sample_data.degrade, seed 41; files *_e41.npz from
prep_data.py --eval). Reported in metres against the CLEAN DFL ball, at 5 Hz samples:
    all / seen / gap frames (gap = degraded ball missing), alive gap frames, gaps > 2 s;
    baselines in gaps: linear interpolation of the observed ball, and the expected possessor
    position (sum of player-level possession probabilities x slot positions);
team head accuracy on alive frames (argmax H vs A) and dead AUC-ish accuracy.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import radar_common as rc  # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")


def interp_baseline(xf, seen):
    t = np.arange(len(seen))
    out = np.zeros((len(seen), 2))
    if seen.sum() < 2:
        return out
    for i in (0, 1):
        out[:, i] = np.interp(t, t[seen], xf[seen, i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tag", default="radar")
    ap.add_argument("--games", type=int, nargs="*", default=None)
    ap.add_argument("--seed_name", default="e41")
    args = ap.parse_args()
    npz = WORK + "/idsse/npz"
    matches = sorted(f[:-4] for f in os.listdir(npz) if f.endswith(".npz"))
    games = args.games if args.games is not None else range(len(matches))
    tot = {}
    for gi in games:
        wpath = os.path.join(args.folds, "fold_{}_{}.npz".format(args.tag, gi))
        if not os.path.isfile(wpath):
            continue
        W = rc.load_weights(wpath)
        z = np.load(os.path.join(args.seq, "{}_{}.npz".format(matches[gi], args.seed_name)))
        rows = {}
        for h in (1, 2):
            inp = {k: z["h{}_{}".format(h, k)] for k in ("xs", "ms", "pos", "xf")}
            o = rc.forward(W, inp)
            os_ = rc.forward(W, rc.swap(inp))
            ball = 0.5 * (o["ball"] - os_["ball"])
            team = 0.5 * (o["team"] + os_["team"][:, [1, 0, 2]])
            gt = z["h{}_ball".format(h)].astype(np.float64)
            code = z["h{}_code".format(h)]
            seen = z["h{}_seen".format(h)] > 0
            lin = interp_baseline(inp["xf"][:, :2].astype(np.float64), seen)
            err = np.sqrt(((ball - gt) ** 2).sum(1)) * rc.SCALE
            el = np.sqrt(((lin - gt) ** 2).sum(1)) * rc.SCALE
            ee = np.sqrt(((o["epos"] - gt) ** 2).sum(1)) * rc.SCALE
            gl = rc.since_last(seen, 10 ** 6) + rc.since_last(seen[::-1], 10 ** 6)[::-1]
            valid = code < 3
            masks = dict(all=valid, seen=valid & seen, gap=valid & ~seen,
                         gap_alive=valid & ~seen & (code < 2), gap_long=valid & ~seen & (gl > 10))
            for k, m in masks.items():
                rows.setdefault(k, []).append((err[m], el[m], ee[m]))
            alive = code < 2
            pred = np.argmax(team[:, :2], 1)
            rows.setdefault("team", []).append((pred[alive] == code[alive]).astype(float))
            rows.setdefault("dead", []).append(((team[:, 2] > 0.5) == (code == 2))[valid].astype(float))
        line = [matches[gi]]
        for k in ("all", "seen", "gap", "gap_alive", "gap_long"):
            e = np.concatenate([r[0] for r in rows[k]])
            l = np.concatenate([r[1] for r in rows[k]])
            p = np.concatenate([r[2] for r in rows[k]])
            tot.setdefault(k, []).append((e, l, p))
            line.append("{} {:.2f} (lin {:.2f}, poss {:.2f}) n={}".format(k, e.mean(), l.mean(), p.mean(), len(e)))
        ta = np.concatenate(rows["team"]).mean()
        da = np.concatenate(rows["dead"]).mean()
        tot.setdefault("team", []).append(ta)
        tot.setdefault("deadacc", []).append(da)
        line.append("team acc {:.3f} dead acc {:.3f}".format(ta, da))
        print(" | ".join(line), flush=True)
    print("MEAN over games:")
    for k in ("all", "seen", "gap", "gap_alive", "gap_long"):
        e = np.concatenate([r[0] for r in tot[k]])
        l = np.concatenate([r[1] for r in tot[k]])
        p = np.concatenate([r[2] for r in tot[k]])
        print("  {:10s} radar {:.2f} m (median {:.2f})  linear-interp {:.2f} m (median {:.2f})  "
              "expected-possessor {:.2f} m".format(k, e.mean(), np.median(e), l.mean(), np.median(l), p.mean()))
    print("  team acc (alive, H vs A) {:.3f}   dead acc {:.3f}".format(np.mean(tot["team"]),
                                                                       np.mean(tot["deadacc"])))


if __name__ == "__main__":
    main()
