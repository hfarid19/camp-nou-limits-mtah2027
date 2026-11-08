#!/usr/bin/env python3
"""All README "Results" tables for the event-defined control labels.

    python -I candidates/control_events/analyse.py [--labels DIR] [--tag NAME]

1. possessions per 90 under raw DFL, control.fold(T) and the event labels;
2. how well control.fold(T) of the DFL labels matches the event labels;
3. every saved bench-v2 prediction scored against the event labels, paired differences;
4. nested leave-one-game-out choice of the fold threshold for ensemble_vote's output;
5. a per-spell keep-or-hand-back logistic rule vs the fixed fold, nested leave-one-game-out.
Writes a JSON of everything to $MTAH_WORK/arch/control_events/analysis<_tag>.json.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_events as S          # noqa: E402  (puts candidates/ and evaluation/ on the path)
import control                    # noqa: E402
import possession_metrics as pm   # noqa: E402

FPS = 25
MATCHES = sorted(f[:-4] for f in os.listdir(S.NPZ) if f.endswith(".npz"))
SEEDS = (2027, 2028, 2029)
PROBS = os.path.join(S.WORK, "probs")
CACHE = os.path.join(S.WORK, "bench_cache")
CANDS = ["ensemble_vote", "ensemble_ctrl", "handover_mstcn", "temporal_tcn", "learned_dfl_evidence",
         "selftrain_ctrl", "decision_tree_b"]
TGRID = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]


def n_spells(codes):
    return sum(1 for s in pm.segments(codes, pm.valid_runs(codes)) if s[2] in (pm.H, pm.A))


def sf1(gt, pred):
    return pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"]


def per90(count, frames):
    return count / (frames / FPS / 60.0) * 90.0


# ---------------------------------------------------------------- 1 + 2: label comparisons
def label_tables(labels):
    rows = {}
    tot_frames = sum(int((S.raw_gt(m) != pm.X).sum()) for m in MATCHES)
    for T in TGRID:
        cnt, agree, f1 = 0, [], []
        for m in MATCHES:
            gt, ev = S.raw_gt(m), S.event_gt(m, labels)
            fo = control.fold(gt, T, pm.valid_runs(gt))
            cnt += n_spells(fo)
            alive = (ev == pm.H) | (ev == pm.A)
            agree.append(float(np.mean(fo[alive] == ev[alive])))
            f1.append(sf1(ev, fo))
        rows[T] = dict(per90=per90(cnt, tot_frames), spells=cnt, frame_agree=float(np.mean(agree)),
                       spellF1_vs_evt=float(np.mean(f1)), spellF1_games=f1)
    ev_cnt = sum(n_spells(S.event_gt(m, labels)) for m in MATCHES)
    return dict(fold=rows, event=dict(per90=per90(ev_cnt, tot_frames), spells=ev_cnt))


# ---------------------------------------------------------------- 3: candidates
def paired(a, b):
    """a, b: {(match, seed): value}; paired over common keys. SE over the 7 game means."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    games = sorted({k[0] for k in keys})
    gm = np.array([np.mean([a[k] - b[k] for k in keys if k[0] == g]) for g in games])
    se = gm.std(ddof=1) / np.sqrt(len(gm)) if len(gm) > 1 else np.nan
    return dict(n=len(keys), mean=float(d.mean()), se_games=float(se), wins=int((d > 0).sum()),
                losses=int((d < 0).sum()))


def candidates(labels):
    dirs = sorted(d for d in os.listdir(PROBS)
                  if any(f.startswith("dfl2_") and f.endswith("_codes.npy")
                         for f in os.listdir(os.path.join(PROBS, d))))
    res = {d: S.score_dir(os.path.join(PROBS, d), labels) for d in dirs}
    table = {}
    for d, r in res.items():
        table[d] = {k: float(np.mean([v[k] for v in r.values()])) for k in ("evt", "evt_macro", "raw", "ctrl_mean")}
        table[d]["n"] = len(r)
        table[d]["poss_per_seed"] = float(np.sum([v["poss"] for v in r.values()]) / max(1, len(r)) * 7)
    pairs = {}
    for ref in ("ensemble_vote", "ensemble_ctrl"):
        for d in CANDS:
            if d in res and d != ref:
                pairs["{} - {}".format(d, ref)] = dict(
                    evt=paired({k: v["evt"] for k, v in res[d].items()}, {k: v["evt"] for k, v in res[ref].items()}),
                    ctrl_mean=paired({k: v["ctrl_mean"] for k, v in res[d].items()},
                                     {k: v["ctrl_mean"] for k, v in res[ref].items()}))
    full = [d for d in dirs if table[d]["n"] == 21]
    from scipy.stats import spearmanr, kendalltau
    rho = spearmanr([table[d]["evt"] for d in full], [table[d]["ctrl_mean"] for d in full])[0]
    tau = kendalltau([table[d]["evt"] for d in full], [table[d]["ctrl_mean"] for d in full])[0]
    return table, pairs, dict(spearman_evt_vs_ctrl_mean=float(rho), kendall=float(tau), n=len(full)), res


# ---------------------------------------------------------------- 4: nested fold threshold
def load_codes(slug, m, s):
    p = os.path.join(PROBS, slug, "dfl2_{}_{}_codes.npy".format(m, s))
    return np.load(p) if os.path.isfile(p) else None


def fold_threshold(labels, slug="ensemble_vote", grid=None):
    grid = grid or [x / 4 for x in range(0, 21)]          # 0 .. 5 s
    score = np.zeros((len(MATCHES), len(grid)))             # mean over seeds
    for i, m in enumerate(MATCHES):
        ev = S.event_gt(m, labels)
        runs = pm.valid_runs(ev)
        for s in SEEDS:
            pred = load_codes(slug, m, s)
            for j, T in enumerate(grid):
                score[i, j] += sf1(ev, control.fold(pred, T, runs)) / len(SEEDS)
    nested = []
    for i in range(len(MATCHES)):
        inner = np.delete(score, i, axis=0).mean(0)
        j = int(np.argmax(inner))
        nested.append((MATCHES[i], grid[j], float(score[i, j])))
    j2 = grid.index(2.0)
    return dict(grid=grid, mean_by_T=score.mean(0).tolist(), best_T_all=grid[int(np.argmax(score.mean(0)))],
                nested=nested, nested_mean=float(np.mean([n[2] for n in nested])),
                fixed2_mean=float(score[:, j2].mean()), nofold_mean=float(score[:, 0].mean()),
                nested_minus_fixed2=paired_games([n[2] for n in nested], score[:, j2]))


def paired_games(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), wins=int((d > 0).sum()),
                losses=int((d < 0).sum()))


# ---------------------------------------------------------------- 5: per-spell rule
def spell_features(pred, probs, ball, slots, slot_team):
    """Contested spells of pred (team X directly after the other team, single left-to-right view
    of the ORIGINAL prediction) and their features. Features use only what the pipeline sees."""
    out = []
    segs = pm.segments(pred, pm.valid_runs(pred))
    for i, (a, b, c) in enumerate(segs):
        if c not in (pm.H, pm.A) or i == 0:
            continue
        pa, pb, pc = segs[i - 1]
        if pc not in (pm.H, pm.A) or pc == c or pb != a:
            continue
        nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == b else None
        dur = (b - a) / FPS
        marg = probs[a:b, c] - probs[a:b, 1 - c]
        vis = ~np.isnan(ball[a:b, 0])
        dx = np.nan
        if vis.any():
            bb = ball[a:b][vis]
            P = slots[a:b][vis]
            d = np.linalg.norm(P - bb[:, None, :], axis=2)
            own = np.nanmin(np.where(slot_team[None, :] == c, d, np.inf), axis=1)
            opp = np.nanmin(np.where(slot_team[None, :] != c, d, np.inf), axis=1)
            ok = np.isfinite(own) & np.isfinite(opp)
            dx = float(np.mean(np.clip(own[ok] - opp[ok], -10, 10))) if ok.any() else np.nan
        out.append(dict(a=a, b=b, c=c, f=[np.log(dur), float(np.mean(marg)), float(np.max(probs[a:b, c])),
                                          float(vis.mean()), 0.0 if np.isnan(dx) else dx, float(np.isnan(dx)),
                                          np.log((pb - pa) / FPS),
                                          float(nxt is not None and nxt[2] == pc),
                                          float(nxt is None or nxt[2] == pm.D)]))
    return out


FEATS = ["log_dur", "mean_margin", "max_p", "ball_visible", "own_minus_opp_dist", "dist_missing",
         "log_prev_dur", "next_is_prev_team", "next_is_dead"]


def apply_rule(pred, spells, keep):
    """Single left-to-right pass: a contested spell not kept is given to the current owner."""
    out = pred.copy()
    kmap = {(s["a"], s["b"]): k for s, k in zip(spells, keep)}
    owner, prev_end = None, None
    for a, b, c in pm.segments(pred, pm.valid_runs(pred)):
        if c not in (pm.H, pm.A):
            owner = None
            continue
        if owner is None or prev_end != a:
            owner = c
        elif c != owner:
            k = kmap.get((a, b), True)
            if k:
                owner = c
            else:
                out[a:b] = owner
        prev_end = b
    return out


def spell_rule(labels, slug="ensemble_vote"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    data = {}
    for m in MATCHES:
        ev = S.event_gt(m, labels)
        for s in SEEDS:
            pred = load_codes(slug, m, s)
            probs = np.load(os.path.join(PROBS, slug, "dfl2_{}_{}.npy".format(m, s)))
            z = np.load(os.path.join(CACHE, "dfl2_{}_{}.npz".format(m, s)))
            sp = spell_features(pred, probs, z["ball"], z["slots"], z["slot_team"])
            y = np.array([float(np.mean(ev[x["a"]:x["b"]] == x["c"]) > 0.5) for x in sp])
            data[(m, s)] = (ev, pred, sp, y)
    thr_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    dur_grid = [x / 4 for x in range(0, 21)]
    out = []
    for hold in MATCHES:
        tr = [k for k in data if k[0] != hold]
        X = np.array([x["f"] for k in tr for x in data[k][2]])
        y = np.concatenate([data[k][3] for k in tr])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)

        def run(keys, thr):
            r = []
            for k in keys:
                ev, pred, sp, _ = data[k]
                p = clf.predict_proba(sc.transform(np.array([x["f"] for x in sp])))[:, 1] if sp else []
                r.append(sf1(ev, apply_rule(pred, sp, [pi >= thr for pi in p])))
            return float(np.mean(r))

        def run_dur(keys, T):
            return float(np.mean([sf1(data[k][0], apply_rule(data[k][1], data[k][2],
                                                             [np.exp(x["f"][0]) >= T for x in data[k][2]]))
                                  for k in keys]))
        thr = max(thr_grid, key=lambda t: run(tr, t))          # threshold chosen on the 6 training games
        Td = max(dur_grid, key=lambda t: run_dur(tr, t))      # same single-pass rule, duration only
        te = [k for k in data if k[0] == hold]
        out.append(dict(game=hold, thr=thr, T_dur=Td, clf=run(te, thr), dur=run_dur(te, Td),
                        fold2=float(np.mean([sf1(data[k][0], control.fold(data[k][1], 2.0, pm.valid_runs(data[k][0])))
                                             for k in te])),
                        coef=dict(zip(FEATS, np.round(clf.coef_[0], 3).tolist()))))
    allX = np.array([x["f"] for k in data for x in data[k][2]])
    ally = np.concatenate([data[k][3] for k in data])
    return dict(folds=out, clf_mean=float(np.mean([o["clf"] for o in out])),
                dur_mean=float(np.mean([o["dur"] for o in out])),
                fold2_mean=float(np.mean([o["fold2"] for o in out])),
                clf_minus_fold2=paired_games([o["clf"] for o in out], [o["fold2"] for o in out]),
                clf_minus_dur=paired_games([o["clf"] for o in out], [o["dur"] for o in out]),
                n_spells=int(len(ally)), keep_rate=float(ally.mean()), n_feat=allX.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=S.LABELS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-rule", action="store_true")
    args = ap.parse_args()
    res = {}
    res["labels"] = label_tables(args.labels)
    print("== possessions per 90 and fold(T) vs event labels")
    print("event labels: {:.0f} per 90 ({} spells)".format(res["labels"]["event"]["per90"], res["labels"]["event"]["spells"]))
    for T, r in res["labels"]["fold"].items():
        print("  fold T={:<4} per90 {:5.0f}  frame agree {:.4f}  spellF1 vs evt {:.4f}".format(
            T, r["per90"], r["frame_agree"], r["spellF1_vs_evt"]))
    table, pairs, rank, _ = candidates(args.labels)
    res.update(candidates=table, pairs=pairs, rank_agreement=rank)
    print("== candidates (mean over game-seeds)")
    for d in sorted(table, key=lambda d: -table[d]["evt"]):
        t = table[d]
        print("  {:22s} n={:2d} evt {:.4f}  evtMacro {:.4f}  ctrl_mean {:.4f}  raw {:.4f}  poss/seed {:5.0f}".format(
            d, t["n"], t["evt"], t["evt_macro"], t["ctrl_mean"], t["raw"], t["poss_per_seed"]))
    print("  rank agreement evt vs ctrl_mean:", rank)
    for k, v in pairs.items():
        print("  {:40s} evt {:+.4f} (SE {:.4f}, {}/{} wins)   ctrl_mean {:+.4f} (SE {:.4f}, {}/{})".format(
            k, v["evt"]["mean"], v["evt"]["se_games"], v["evt"]["wins"], v["evt"]["n"],
            v["ctrl_mean"]["mean"], v["ctrl_mean"]["se_games"], v["ctrl_mean"]["wins"], v["ctrl_mean"]["n"]))
    res["fold_threshold"] = {}
    for slug in ("ensemble_vote", "handover_mstcn", "temporal_tcn"):
        ft = fold_threshold(args.labels, slug)
        res["fold_threshold"][slug] = ft
        print("== fold threshold on {} output vs event labels".format(slug))
        print("  mean spellF1 by T:", " ".join("{}:{:.4f}".format(T, v) for T, v in zip(ft["grid"], ft["mean_by_T"])))
        print("  best T on all 7: {}; nested LOGO picks {}; nested {:.4f} vs fixed 2.0 {:.4f} vs no fold {:.4f}; "
              "nested - fixed2 {}".format(ft["best_T_all"], [n[1] for n in ft["nested"]], ft["nested_mean"],
                                         ft["fixed2_mean"], ft["nofold_mean"], ft["nested_minus_fixed2"]))
    if not args.skip_rule:
        sr = spell_rule(args.labels)
        res["spell_rule"] = sr
        print("== per-spell logistic keep/hand-back rule (nested LOGO) on ensemble_vote")
        for o in sr["folds"]:
            print("  ", o["game"], "thr", o["thr"], "T_dur", o["T_dur"], "clf {:.4f} dur {:.4f} fold2 {:.4f}".format(
                o["clf"], o["dur"], o["fold2"]))
        print("  clf {:.4f}  single-pass duration {:.4f}  control.fold 2.0 {:.4f}".format(
            sr["clf_mean"], sr["dur_mean"], sr["fold2_mean"]))
        print("  clf - fold2", sr["clf_minus_fold2"], " clf - dur", sr["clf_minus_dur"])
        print("  coefficients (last fold):", sr["folds"][-1]["coef"])
    out = os.path.join(S.WORK, "arch", "control_events", "analysis{}.json".format("_" + args.tag if args.tag else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
