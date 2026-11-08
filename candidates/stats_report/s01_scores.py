"""Per game-seed scores of every candidate with saved bench-v2 codes, against labels folded at many T.

Writes $STATS/scores.csv (one row per candidate x game x seed x T: spellF1, macroF1, n_pred, n_gt)
and $STATS/games.csv (label-derived and label-free characteristics of each game-seed).

Candidates: every probs/<slug> with 21 dfl2 codes files, decision_tree/raw (no fold), the rebuilt
selftrain_ctrl (s00), and derived variants "<slug>+fold<t>" = control.fold applied to the
candidate's own output (ensemble_vote at t = 1.0 ... 3.0; learned_dfl_evidence, temporal_tcn,
handover_mstcn and decision_tree/raw at 2.0). ball_radar and semimarkov_lattice saved no codes;
their raw (T = 0) values are read from rescore_v2/<slug>.txt (no ctrl values were saved).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from common import PROBS, STATS, WORK, MATCHES, SEEDS, bench, pm, control, runs_from_blocks

TS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
FOLD_EV = [1.0, 1.5, 2.0, 2.5, 3.0]


def codes_dirs():
    d = {}
    for p in sorted(glob.glob(os.path.join(PROBS, "*"))):
        if len(glob.glob(os.path.join(p, "dfl2_*_codes.npy"))) == 21:
            d[os.path.basename(p)] = p
    raw = os.path.join(PROBS, "decision_tree", "raw")
    if len(glob.glob(os.path.join(raw, "dfl2_*_codes.npy"))) == 21:
        d["decision_tree_raw"] = raw
    d["selftrain_ctrl"] = os.path.join(STATS, "selftrain_codes")
    return d


def main():
    dirs = codes_dirs()
    rows, games = [], []
    for m in MATCHES:
        for seed in SEEDS:
            slots, st, ball, gt, blocks, sign = bench.dfl2_prepared(m, seed)
            runs = runs_from_blocks(blocks)
            graded = gt != pm.X
            labs = {T: bench.controlled_labels(gt, T) for T in TS}
            mins = graded.sum() / 25 / 60
            seg = [(a, b, c) for a, b, c in pm.segments(gt, runs) if c in (pm.H, pm.A)]
            seg2 = [(a, b, c) for a, b, c in pm.segments(labs[2], runs) if c in (pm.H, pm.A)]
            games.append(dict(game=m, seed=seed, minutes=mins,
                              dead_share=float((gt[graded] == pm.D).mean()),
                              ball_missing=float(np.isnan(ball[graded, 0]).mean()),
                              ball_missing_alive=float(np.isnan(ball[graded & (gt != pm.D), 0]).mean()),
                              ball_missing_dead=float(np.isnan(ball[gt == pm.D, 0]).mean()),
                              poss_per90_raw=len(seg) * 90 / mins, poss_per90_T2=len(seg2) * 90 / mins,
                              mean_spell_s=float(np.mean([b - a for a, b, _ in seg]) / 25),
                              short_share_lt2s=float(np.mean([(b - a) < 50 for a, b, _ in seg])),
                              home_share=float((gt[graded] == pm.H).sum() / max(1, (gt[graded] < 2).sum()))))
            for slug, d in dirs.items():
                pred = np.load(os.path.join(d, "dfl2_{}_{}_codes.npy".format(m, seed)))
                variants = {slug: pred}
                if slug == "ensemble_vote":
                    for t in FOLD_EV:
                        variants["ensemble_vote+fold{:g}".format(t)] = control.fold(pred, t, runs)
                if slug in ("learned_dfl_evidence", "temporal_tcn", "handover_mstcn", "decision_tree_raw"):
                    variants[slug + "+fold2"] = control.fold(pred, 2.0, runs)
                for name, p in variants.items():
                    for T in TS:
                        r = pm.evaluate(labs[T], p, bootstrap_n=0)
                        rows.append(dict(cand=name, game=m, seed=seed, T=T, spellF1=r["spell_f1_mean"],
                                         macroF1=r["macro_f1"], n_pred=r["n_pred_spells"],
                                         n_gt=r["n_gt_spells"]))
            print(m, seed, len(rows), flush=True)
    for slug in ("ball_radar", "semimarkov_lattice"):
        txt = open(os.path.join(WORK, "rescore_v2", slug + ".txt")).read()
        res = json.loads(txt[txt.rindex("RESULT ") + 7:].splitlines()[0])
        for r in res["rows"]:
            rows.append(dict(cand=slug, game=r["game"], seed=r["seed"], T=0, spellF1=r["spellF1"],
                             macroF1=r["macroF1"], n_pred=int(r["poss"].split("/")[0]),
                             n_gt=int(r["poss"].split("/")[1])))
            for T, v in zip((1, 2, 3), r.get("ctrl", [])):
                rows.append(dict(cand=slug, game=r["game"], seed=r["seed"], T=T, spellF1=v,
                                 macroF1=np.nan, n_pred=np.nan, n_gt=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS, "scores.csv"), index=False)
    pd.DataFrame(games).to_csv(os.path.join(STATS, "games.csv"), index=False)
    # sanity: ensemble_ctrl == ensemble_vote + fold 2
    a = df[(df.cand == "ensemble_ctrl")].set_index(["game", "seed", "T"]).spellF1
    b = df[(df.cand == "ensemble_vote+fold2")].set_index(["game", "seed", "T"]).spellF1
    print("max |ensemble_ctrl - ensemble_vote+fold2| =", float((a - b).abs().max()))
    cm = df[df["T"].isin([1, 2, 3])].groupby("cand").spellF1.mean().sort_values(ascending=False)
    print(cm.round(4).to_string())


if __name__ == "__main__":
    main()
