#!/usr/bin/env python3
"""Fit degrade_v4's dead-ball knobs to the organisers' published full-match baseline scores and
to the match's label-free ball-run statistics.

    python candidates/fit_degrade_v4.py --evals 200          # random search, prints FITTED dict
    python candidates/fit_degrade_v4.py --check [params.json] # score DEFAULTS (or a dict) on 7 games

Targets (no ground truth of the match is read anywhere):
- EVALUATION.md: full-match macroF1 / spellF1 of starter/baseline.py's four variants (nearest
  player to the ball, D where the ball is missing, majority vote over +-0 / 25 / 50 / 125 frames):
  0.6532/0.0486, 0.6752/0.1996, 0.6724/0.2433, 0.6356/0.2242. The same baselines are run on the
  degraded DFL games; spellF1 is read against the fine4 label readings (score_sel2.FINE), since
  the published shift fingerprint puts the truth between the raw DFL flag and a 1 s fold.
- match_stats-like run statistics of the match ball (v5b_calib3/4): gap and visible-run length
  histograms, slow-island share, ball speed in slow islands, out-runs per 90, ball missing.
Ball knobs only: degrade_v2's player knobs are kept (the players are degraded once per game and
reused). Fitted on the first 3 DFL games (1 seed), checked on all 7.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
sys.path.insert(0, HERE)
import bench                        # noqa: E402
import degrade_v2 as dv             # noqa: E402
import degrade_v4 as d4             # noqa: E402
import possession_metrics as pm     # noqa: E402
import possession_pipeline as pp    # noqa: E402

FPS = 25
PUB = {0: (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}
FINE_T = (0.5, 1.0)
EVT1 = os.path.join(bench.WORK, "arch", "control_events", "variants", "hold1")
SPACE = dict(roll_s=(0.0, 1.5), p_roll=(0.0, 1.0), reveal_s=(0.0, 10.0), park_s=(0.0, 3.0),
             isl_per_min=(0.0, 6.0), isl_s=(0.8, 8.0), isl_sigma=(0.3, 1.5), isl_first_s=(0.0, 15.0),
             isl_offset_m=(0.3, 3.0), isl_step_m=(0.0, 0.3),
             alive_gaps_per90=(0.0, 10.0), alive_gap_min_s=(0.0, 6.0), alive_gap_s=(0.5, 6.0))
# alive gaps are capped at 10 per 90: by gap geometry and player speed the match has about 5 per 90
# gaps >= 5 s that look like open play (v5b_calib7.py); the first fit, uncapped, asked for 40.
# match run statistics (candidates/_scratch/v5b_calib1/3/4.py on BAR-ATH, label-free)
MATCH = {"ball_missing": 0.407, "gap_0.5-5": 1.7, "gap_5-10": 45.0, "gap_10-30": 51.0, "gap_30+": 16.1,
         "vis_1-2": 9.3, "vis_2-5": 17.0, "vis_5-10": 22.9, "vis_10-30": 42.5, "slow_isl_share": 0.517,
         "slow_isl_ball_spd": 2.84, "slow_isl_near_d": 1.70, "out_runs": 68.8, "out_run_p50": 0.84,
         "static_share": 0.050}
MATCH_W = {"ball_missing": 3.0, "gap_0.5-5": 0.3, "gap_5-10": 0.7, "gap_10-30": 0.7, "gap_30+": 0.5,
           "vis_1-2": 0.5, "vis_2-5": 0.5, "vis_5-10": 0.5, "vis_10-30": 0.3, "slow_isl_share": 1.0,
           "slow_isl_ball_spd": 0.5, "slow_isl_near_d": 0.5, "out_runs": 0.5, "out_run_p50": 0.3,
           "static_share": 1.0}


def load_games(names, seed=7):
    games = []
    for i, m in enumerate(names):
        z = np.load(os.path.join(bench.DFL_DIR, m + ".npz"))
        status, poss, half = z["status"], z["poss"], z["half"]
        gt = np.full(len(status), pm.X, np.int8)
        gt[(status == 1) & (poss == 1)] = pm.H
        gt[(status == 1) & (poss == 2)] = pm.A
        gt[status == 0] = pm.D
        blocks = []
        for h in (1, 2):
            idx = np.where((half == h) & (gt != pm.X))[0]
            blocks.append((len(blocks) + 1, int(idx[0]), int(idx[-1])))
        players = z["players"].astype(np.float64)
        slots, slot_team = dv.degrade_players(players, z["team_of_player"].astype(int),
                                              np.random.default_rng(seed + i), dv.DEFAULTS)
        graded = gt != pm.X
        labels = {"raw": gt}
        for T in FINE_T:
            labels["T%g" % T] = bench.controlled_labels(gt, T)
        labels["evt1"] = np.load(os.path.join(EVT1, m + ".npy"))
        games.append(dict(name=m, players=players, ball=z["ball"].astype(np.float64), status=status,
                          graded=graded, blocks=blocks, slots=slots, slot_team=slot_team, gt=gt,
                          labels=labels, team_speed=team_speed(players)))
    return games


def team_speed(players):
    v = np.linalg.norm(np.diff(players, axis=0), axis=2) * FPS
    v = np.where(v > 12, np.nan, v)
    return np.concatenate([[np.nan], np.nanmean(v, axis=1)])


def nearest_team(players, slot_team, ball):
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(1)
    best = d[np.arange(len(d)), k]
    return np.where(np.isfinite(best), slot_team[k], -1).astype(np.int8)


def smooth(codes, window):
    if window <= 0:
        return np.where(codes == 0, 0, np.where(codes == 1, 1, 2)).astype(np.int8)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(codes), idx + window + 1)
    h = hc[hi] - hc[lo]
    a = ac[hi] - ac[lo]
    lab = np.full(len(codes), 2, np.int8)
    seen = (h + a) > 0
    lab[seen] = np.where(h[seen] >= a[seen], 0, 1)
    return lab


def baseline_scores(g, ball):
    """{window: (macroF1 vs raw flag, mean spellF1 over fine4 readings)}"""
    nt = nearest_team(g["slots"], g["slot_team"], ball)
    out = {}
    for w in PUB:
        pred = smooth(nt, w)
        pred[~g["graded"]] = pm.X
        mf = pm.evaluate(g["gt"], pred, bootstrap_n=0)["macro_f1"]
        sf = np.mean([pm.evaluate(lab, pred, bootstrap_n=0)["spell_f1_mean"] for lab in g["labels"].values()])
        out[w] = (mf, sf)
    return out


def run_stats(g, ball, pitch=bench.tsd.PITCH):
    L, Wd = pitch
    gr = g["graded"]
    seen = ~np.isnan(ball[:, 0])
    ms = g["team_speed"]
    mins = gr.sum() / FPS / 60
    per90 = 90 / mins
    out = {"ball_missing": float((~seen & gr).sum() / gr.sum())}
    s, e = pp.runs_of(~seen & gr)
    Ls = (e - s) / FPS
    out["gap_0.5-5"] = ((Ls >= 0.5) & (Ls < 5)).sum() * per90
    out["gap_5-10"] = ((Ls >= 5) & (Ls < 10)).sum() * per90
    out["gap_10-30"] = ((Ls >= 10) & (Ls < 30)).sum() * per90
    out["gap_30+"] = (Ls >= 30).sum() * per90
    s, e = pp.runs_of(seen & gr)
    Ls = (e - s) / FPS
    out["vis_1-2"] = ((Ls >= 1) & (Ls < 2)).sum() * per90
    out["vis_2-5"] = ((Ls >= 2) & (Ls < 5)).sum() * per90
    out["vis_5-10"] = ((Ls >= 5) & (Ls < 10)).sum() * per90
    out["vis_10-30"] = ((Ls >= 10) & (Ls < 30)).sum() * per90
    sel = (Ls >= 1) & (Ls < 10)
    spd = np.array([np.nanmean(ms[a:b]) for a, b in zip(s[sel], e[sel])])
    slow = spd < 1.5
    out["slow_isl_share"] = float(slow.mean()) if len(slow) else 0.0
    bv = np.concatenate([[np.nan], np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS])
    d = np.sqrt(((g["slots"] - ball[:, None, :]) ** 2).sum(-1))
    nd = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    sb = [np.nanmedian(bv[a:b]) for a, b, sl in zip(s[sel], e[sel], slow) if sl]
    sd = [np.nanmedian(nd[a:b]) for a, b, sl in zip(s[sel], e[sel], slow) if sl]
    out["slow_isl_ball_spd"] = float(np.nanmedian(sb)) if sb else 0.0
    out["slow_isl_near_d"] = float(np.nanmedian(sd)) if sd else 0.0
    o = (ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > Wd)
    s, e = pp.runs_of(o & seen & gr)
    out["out_runs"] = len(s) * per90
    out["out_run_p50"] = float(np.median((e - s) / FPS)) if len(s) else 0.0
    out["static_share"] = float(np.nanmean(bv[seen & gr] < 0.5))
    return out


def evaluate(games, p, seed=11):
    bs, rs = [], []
    for i, g in enumerate(games):
        ball = d4.dead_ball(g["ball"], g["players"], g["status"], g["graded"], np.random.default_rng(seed + i), p)
        bs.append(baseline_scores(g, ball))
        rs.append(run_stats(g, ball))
    base = {w: (float(np.mean([b[w][0] for b in bs])), float(np.mean([b[w][1] for b in bs]))) for w in PUB}
    stats = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    return base, stats


STRUCTURE_ONLY = False


def distance(base, stats, w_stats=1.0):
    ds = sum(MATCH_W[k] * abs(stats[k] - MATCH[k]) / max(abs(MATCH[k]), 1e-6) for k in MATCH)
    if STRUCTURE_ONLY:
        return ds, 0.0, ds
    d = 0.3 * abs(base[0][0] - PUB[0][0]) / 0.02
    for w, (mf, sf) in PUB.items():
        d += abs(base[w][1] - sf) / 0.02
        if w:
            d += abs((base[w][0] - base[0][0]) - (mf - PUB[0][0])) / 0.02
    ds = sum(MATCH_W[k] * abs(stats[k] - MATCH[k]) / max(abs(MATCH[k]), 1e-6) for k in MATCH)
    return d + w_stats * ds, d, ds


def report(base, stats):
    print("  baseline   bench macroF1 spellF1 | published")
    for w in PUB:
        print("  +-{:<4d} {:.4f} {:.4f} | {:.4f} {:.4f}".format(w, base[w][0], base[w][1], *PUB[w]))
    print("  " + "  ".join("{} {:.2f}/{:.2f}".format(k, stats[k], MATCH[k]) for k in MATCH))


FIXED = dict(isl_per_min=0.45,     # ~18 extra gaps per 90 (match 112 gaps >= 5 s vs 96 DFL stoppages)
             roll_s=0.8, p_roll=0.85,  # out-runs: 69 per 90, median 0.84 s
             park_s=0.8,              # static ball 5.0% of visible frames vs 2.4% in clean alive play
             alive_gaps_per90=5.0, alive_gap_min_s=4.5, alive_gap_s=2.0)   # v5b_calib7 geometry
FREE = ("reveal_s", "isl_s", "isl_sigma", "isl_first_s", "isl_offset_m", "isl_step_m")


def sample(rng, best=None, scale=1.0):
    p = {}
    for k in FREE:
        lo, hi = SPACE[k]
        if best is None:
            p[k] = float(rng.uniform(lo, hi))
        else:
            p[k] = float(min(hi, max(lo, best[k] + rng.normal(0, scale * (hi - lo) / 2))))
    return p


def search(games, evals, rng, base_p, w_stats):
    """Global random sample over the free knobs (half the budget), then local refinement."""
    results = []
    for i in range(evals // 2):
        p = dict(base_p, **FIXED, **sample(rng))
        d, db, ds = distance(*evaluate(games, p), w_stats)
        results.append((d, p))
        if d <= min(r[0] for r in results):
            print("  global {:3d}  dist {:.3f} (base {:.2f} stats {:.2f})  {}".format(
                i, d, db, ds, {k: round(p[k], 3) for k in FREE}), flush=True)
    results.sort(key=lambda r: r[0])
    best_d, best_p = results[0]
    for i in range(evals - evals // 2):
        scale = 0.3 if i < evals // 4 else 0.1
        p = dict(best_p, **sample(rng, best_p, scale))
        d, db, ds = distance(*evaluate(games, p), w_stats)
        if d < best_d:
            best_p, best_d = p, d
            print("  local  {:3d}  dist {:.3f} (base {:.2f} stats {:.2f})  {}".format(
                i, d, db, ds, {k: round(p[k], 3) for k in FREE}), flush=True)
    return best_p, best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument("--check", nargs="?", const="", default=None, metavar="PARAMS_JSON")
    ap.add_argument("--w-stats", type=float, default=1.0)
    ap.add_argument("--fit-games", type=int, default=3)
    ap.add_argument("--structure-only", action="store_true",
                    help="fit the label-free run statistics only; the baseline scores are reported, not fitted")
    ap.add_argument("--free-all", action="store_true", help="search every knob in SPACE (no FIXED ones)")
    args = ap.parse_args()
    global STRUCTURE_ONLY, FREE
    STRUCTURE_ONLY = args.structure_only
    if args.free_all:
        FREE = tuple(SPACE)
        FIXED.clear()
    names = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    if args.check is not None:
        p = dict(d4.DEFAULTS)
        if args.check:
            p.update(json.load(open(args.check)))
        games = load_games(names)
        base, stats = evaluate(games, p)
        print("check on {} games: dist {:.3f} (base {:.2f} stats {:.2f})".format(len(games), *distance(base, stats, args.w_stats)))
        report(base, stats)
        return
    games = load_games(names[:args.fit_games])
    rng = np.random.default_rng(2027)
    p, d = search(games, args.evals, rng, dict(d4.DEFAULTS), args.w_stats)
    print("FITTED", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}))
    print("fit games:")
    report(*evaluate(games, p))
    print("all 7 games:")
    report(*evaluate(load_games(names), p))


if __name__ == "__main__":
    main()
