#!/usr/bin/env python3
"""Offline fit of ensemble_ctrl_v3b's spell classifier, with nested leave-one-game-out selection.

    OMP_NUM_THREADS=2 python candidates/ensemble_ctrl_v3b/train.py [--workers 4]

Data: the shipped vote's unfolded LOGO codes ($MTAH_WORK/probs/_sweep/fold0), the members' LOGO
codes and P(H, A, D), and the degraded inputs (bench.dfl2_prepared), 7 DFL games x 3 seeds.

Table: every contested spell of the unfolded vote, features from spellctl (inputs and model outputs
only). Target of a spell of team c (previous team p) under a label reading: 1 if the reading gives
more of the spell's frames to c than to p. Variants (training target):
  evt3    DFL event labels, HOLD 3 s
  evt2    DFL event labels, HOLD 2 s
  evtavg  mean of the two (soft target, cross-entropy)           <- declared primary before scoring
  sel5    mean over the five sel2 readings (T = 1, 1.5, 2, evt3, evt2)
  delta   (added after the first run) keep = 1 if handing back this spell alone, in the unfolded
          vote, lowers the mean spellF1 over the five sel2 readings; weight |delta sel2|
LightGBM, 200 trees, 7 leaves, cross-entropy objective.

Nesting. For each scored game g (outer), the model is trained on the other 6 games (all seeds).
Its threshold on P(keep) and the pass mode (single pass / repeat until stable) are chosen on those
6 games only, each scored by an inner model trained on the remaining 5. So nothing about g enters
g's model, threshold or mode. "joint" also chooses the variant inside the inner loop.

Writes
  $MTAH_WORK/probs/ensemble_ctrl_v3b_<variant>_<mode>/   nested codes (threshold chosen on 6)
  $MTAH_WORK/probs/ensemble_ctrl_v3b_joint/              nested codes, variant also chosen on 6
  $MTAH_WORK/probs/ensemble_ctrl_v3b_durfold/            nested choice of a fixed fold T (baseline)
  $MTAH_WORK/arch/ensemble_ctrl_v3b/fold_spell_<gi>.txt  outer model of the joint choice for game gi
  $MTAH_WORK/arch/ensemble_ctrl_v3b/fold_cfg_<gi>.json   its threshold / mode
  $MTAH_WORK/arch/ensemble_ctrl_v3b/folds.json           ensemble_ctrl_v2's folds + these two files
  $MTAH_WORK/arch/ensemble_ctrl_v3b/grid.json            inner / outer score grids
  candidates/ensemble_ctrl_v3b/model.txt, config.json    full model (all 7 games), LOGO choice on 7
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data                       # noqa: E402
import fastscore                  # noqa: E402
import spellctl as sc             # noqa: E402
import control                    # noqa: E402

VARIANTS = ["evt3", "evt2", "evtavg", "sel5", "delta"]
MODES = ["single", "iter"]
THRS = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
FOLDS_T = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
SEL2 = ("T1", "T1.5", "T2", "evt3", "evt2")
PARAMS = dict(objective="cross_entropy", num_leaves=7, learning_rate=0.05, min_data_in_leaf=30,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=7,
              deterministic=True, force_row_wise=True, num_threads=1, verbose=-1)
ROUNDS = 200
ART = os.path.join(data.bench.WORK, "arch", "ensemble_ctrl_v3b")
PROBS = data.W

G = {}            # (match, seed) -> dict, filled before the workers fork


def load_all():
    for match in data.MATCHES:
        for seed in data.SEEDS:
            F, base, blocks, gt = data.game(match, seed)
            tg = data.targets(match, gt)
            spells, X = sc.table(F, base, blocks)
            y = {}
            for k in SEL2:
                t = tg[k]
                y[k] = np.array([float((t[a:b] == c).mean() > (t[a:b] == 1 - c).mean())
                                 for a, b, c in spells])
            Y = {"evt3": y["evt3"], "evt2": y["evt2"], "evtavg": (y["evt3"] + y["evt2"]) / 2,
                 "sel5": np.mean([y[k] for k in SEL2], 0)}
            T = {k: fastscore.Target(v) for k, v in tg.items()}
            s0 = {k: T[k].f1(base)[0] for k in SEL2}
            dl = []
            for a, b, c in spells:
                cc = base.copy()
                cc[a:b] = cc[a - 1]
                dl.append(np.mean([T[k].f1(cc)[0] - s0[k] for k in SEL2]))
            dl = np.array(dl)
            Y["delta"] = (dl <= 0).astype(float)
            Wt = {"delta": np.abs(dl) + 1e-5}
            G[(match, seed)] = dict(F=F, base=base, blocks=blocks, X=X, spells=spells, Y=Y, W=Wt, T=T)
            print("loaded", match, seed, len(spells), flush=True)


def sel2(key, codes):
    T = G[key]["T"]
    return float(np.mean([T[k].f1(codes)[0] for k in SEL2]))


def train(games, variant):
    import lightgbm as lgb
    keys = [k for k in G if k[0] in games]
    X = np.concatenate([G[k]["X"] for k in keys])
    y = np.concatenate([G[k]["Y"][variant] for k in keys])
    w = None
    if variant in G[keys[0]]["W"]:
        w = np.concatenate([G[k]["W"][variant] for k in keys])
        w = w / w.mean()
    ds = lgb.Dataset(X, y, weight=w, feature_name=list(sc.FEATURES), free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def run_codes(model, key, thr, mode):
    g = G[key]
    return sc.apply(g["F"], g["base"], g["blocks"], model.predict, thr, iterate=(mode == "iter"))


def grid_scores(model, games):
    """{(mode, thr): [sel2 per game-seed]} of model applied to every seed of games."""
    out = {}
    for match in games:
        for seed in data.SEEDS:
            key = (match, seed)
            p1 = model.predict(G[key]["X"])          # pass-1 table is fixed: single mode is cheap
            for mode in MODES:
                for thr in THRS:
                    if mode == "single":
                        c = G[key]["base"].copy()
                        for (a, b, cc), q in zip(G[key]["spells"], p1):
                            if q < thr and c[a - 1] in (0, 1) and c[a - 1] != cc:
                                c[a:b] = c[a - 1]
                    else:
                        c = run_codes(model, key, thr, mode)
                    out.setdefault((mode, thr), []).append(sel2(key, c))
    return out


def best(grid):
    """argmax over (mode, thr) of the mean."""
    return max(grid, key=lambda k: (np.mean(grid[k]), -abs(k[1] - 0.5)))


def worker(gi):
    g = data.MATCHES[gi]
    rest = [m for m in data.MATCHES if m != g]
    inner = {}                                      # variant -> {(mode, thr): scores over 6 games}
    for v in VARIANTS:
        acc = {}
        for h in rest:
            m = train([x for x in rest if x != h], v)
            for k, s in grid_scores(m, [h]).items():
                acc.setdefault(k, []).extend(s)
        inner[v] = acc
    outer_models = {v: train(rest, v) for v in VARIANTS}
    outer = {v: grid_scores(outer_models[v], [g]) for v in VARIANTS}
    # nested choices
    choice = {v: best(inner[v]) for v in VARIANTS}
    jv = max(VARIANTS, key=lambda v: np.mean(inner[v][choice[v]]))
    codes = {}
    for v in VARIANTS:
        for mode in MODES:
            thr = best({k: s for k, s in inner[v].items() if k[0] == mode})[1]
            for seed in data.SEEDS:
                codes[(v, mode, seed)] = (thr, run_codes(outer_models[v], (g, seed), thr, mode))
    for seed in data.SEEDS:
        mode, thr = choice[jv]
        codes[("joint", "", seed)] = (thr, run_codes(outer_models[jv], (g, seed), thr, mode))
    # duration-only baseline: fold T chosen on the 6 games
    dur = {}
    for T in FOLDS_T:
        for h in rest:
            for seed in data.SEEDS:
                k = (h, seed)
                runs = [(f, l + 1) for _, f, l in G[k]["blocks"]]
                dur.setdefault(T, []).append(sel2(k, control.fold(G[k]["base"], T, runs, 25)))
    Tstar = max(FOLDS_T, key=lambda T: np.mean(dur[T]))
    for seed in data.SEEDS:
        k = (g, seed)
        runs = [(f, l + 1) for _, f, l in G[k]["blocks"]]
        codes[("durfold", "", seed)] = (Tstar, control.fold(G[k]["base"], Tstar, runs, 25))
    ser = lambda d: {"{}|{}".format(*k): float(np.mean(s)) for k, s in d.items()}
    return dict(gi=gi, match=g, inner={v: ser(inner[v]) for v in VARIANTS},
                outer={v: {"{}|{}".format(*k): s for k, s in outer[v].items()} for v in VARIANTS},
                choice={v: list(choice[v]) for v in VARIANTS}, joint=[jv] + list(choice[jv]),
                Tstar=Tstar, codes=codes,
                models={v: outer_models[v].model_to_string() for v in VARIANTS})


def sha(path):
    with open(path, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    load_all()
    os.makedirs(ART, exist_ok=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        res = pool.map(worker, range(len(data.MATCHES)))
    # nested codes
    for r in res:
        for (v, mode, seed), (thr, c) in r["codes"].items():
            d = os.path.join(PROBS, "ensemble_ctrl_v3b_" + (v if not mode else v + "_" + mode))
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, data.name(r["match"], seed) + "_codes.npy"), np.asarray(c, np.int8))
    # fold models / configs of the joint nested choice
    for r in res:
        jv, mode, thr = r["joint"]
        with open(os.path.join(ART, "fold_spell_{}.txt".format(r["gi"])), "w") as h:
            h.write(r["models"][jv])
        with open(os.path.join(ART, "fold_cfg_{}.json".format(r["gi"])), "w") as h:
            json.dump(dict(variant=jv, mode=mode, thr=thr), h, indent=1)
    # full-data choice: LOGO (outer) grid summed over the 7 games
    full = {}
    for v in VARIANTS:
        for k in res[0]["outer"][v]:
            full[(v, k)] = float(np.mean([s for r in res for s in r["outer"][v][k]]))
    fv, fk = max(full, key=lambda vk: (full[vk], -abs(float(vk[1].split("|")[1]) - 0.5)))
    fmode, fthr = fk.split("|")[0], float(fk.split("|")[1])
    model = train(data.MATCHES, fv)
    model.save_model(os.path.join(HERE, "model.txt"))
    with open(os.path.join(HERE, "config.json"), "w") as h:
        json.dump(dict(variant=fv, mode=fmode, thr=fthr), h, indent=1)
    imp = dict(zip(sc.FEATURES, model.feature_importance("gain").tolist()))
    folds = json.load(open(os.path.join(data.bench.WORK, "arch", "ensemble_ctrl_v2", "folds.json")))
    folds[sha(os.path.join(HERE, "model.txt"))[:12]] = [ART, "spell"]
    folds[sha(os.path.join(HERE, "config.json"))[:12]] = [ART, "cfg"]
    json.dump(folds, open(os.path.join(ART, "folds.json"), "w"), indent=1)
    json.dump(dict(per_game=[{k: r[k] for k in ("gi", "match", "inner", "outer", "choice", "joint", "Tstar")}
                             for r in res],
                   full_choice=[fv, fmode, fthr], full_grid={"{}|{}".format(*k): v for k, v in full.items()},
                   importance=imp),
              open(os.path.join(ART, "grid.json"), "w"), indent=1)
    print("joint nested choices:", [(r["match"], r["joint"]) for r in res])
    print("duration-only nested T:", [(r["match"], r["Tstar"]) for r in res])
    print("full-data choice:", fv, fmode, fthr, "LOGO sel2 {:.4f}".format(full[(fv, fk)]))
    for v in VARIANTS:
        for mode in MODES:
            ks = [k for k in res[0]["outer"][v] if k.startswith(mode)]
            bk = max(ks, key=lambda k: np.mean([s for r in res for s in r["outer"][v][k]]))
            print("in-sample best (LOGO models, thr on 7) {:7s} {:6s} thr {}  sel2 {:.4f}".format(
                v, mode, bk.split("|")[1], np.mean([s for r in res for s in r["outer"][v][bk]])))
    print("importance (gain, full model):")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:20]:
        print("  {:20s} {:10.1f}".format(k, v))


if __name__ == "__main__":
    main()
