"""Analysis 5: domain shift between the competition match (BAR-ATH, positions only) and DFL.

Never reads a label of the match. Three DFL references:
  train = ds_v2 rows (seed 41, randomised degrade_v2), bench = bench v2 (dfl2_prepared, seed 2027),
  and DFL game-to-game variation (each DFL game vs the other six) as the yardstick for "large".
1. LDE features (stride 5 on graded frames) per feature: KS statistic match vs bench and vs train,
   and the 7 leave-one-game-out KS values within bench. Autocorrelation is handled by thinning
   to one row every 30 s (150 rows; above the ~20-40 s integrated autocorrelation times of
   analysis 1) before computing the KS and Anderson-Darling p-values; BH over features.
2. match_stats label-free tracking statistics: match vs the 21 bench-v2 and 21 training game-seeds
   (z-score and range).
3. Classifier two-sample test (HistGradientBoosting, NaN-native): match vs bench rows, 5-fold
   grouped CV with 5-minute chunks as groups; reference = each DFL game vs the other six with the
   same protocol. AUC 0.5 = indistinguishable.
4. Does it matter? Overlap of shifted features with the features the models use; label-free model
   output statistics on the match vs on bench v2; members' disagreement as a label-free error proxy.
Outputs: $STATS/shift_features.csv, $STATS/shift_trackstats.csv, $STATS/shift_c2st.csv,
$STATS/shift_outputs.csv, fig/shift_ks.png, fig/disagreement.png
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import (STATS, WORK, FIG, CAND, REPO, MATCHES, SEEDS, PROBS, SHORT, bench, pm, plt_setup,
                    C1, C2, C3, GREY, INK2)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(CAND, "learned_dfl_evidence"))
import features as fe  # noqa: E402
import match_stats  # noqa: E402

THIN = 150          # rows of 0.2 s -> one row per 30 s


def graded_mask(n, blocks):
    g = np.zeros(n, bool)
    for _, a, b in blocks:
        g[a:b + 1] = True
    return g


def match_raw():
    import possession_pipeline as pp
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    slot_team, _, _, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    return players, slot_team, ball, sign_half1, half_video_blocks(GAME), GAMES[GAME]["pitch"], GAMES[GAME]["fps"]


def feats_cached(tag, fn):
    path = os.path.join(STATS, "feat_{}.npz".format(tag))
    if not os.path.isfile(path):
        players, slot_team, ball, sign_half1, blocks, pitch, fps = fn()
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        idx = np.where(graded_mask(len(ball), blocks))[0][::5]
        np.savez(path, X=F[idx], idx=idx, names=np.array(names), kinds=np.array(kinds))
    z = np.load(path, allow_pickle=True)
    return z["X"].astype(np.float64), list(z["names"])


def bench_raw(m, seed=2027):
    slots, st, ball, gt, blocks, sign = bench.dfl2_prepared(m, seed)
    return slots, st, ball, sign, blocks, bench.tsd.PITCH, 25


def ks_p(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    r = stats.ks_2samp(a, b)
    return r.statistic, r.pvalue


def part_features():
    Xm, names = feats_cached("match", match_raw)
    Xb = {m: feats_cached("bench_" + m, lambda m=m: bench_raw(m))[0] for m in MATCHES}
    Xt = {m: np.load(os.path.join(WORK, "ds_v2/ds/{}_41.npz".format(m)), allow_pickle=True)["X"].astype(np.float64)
          for m in MATCHES}
    XB = np.vstack(list(Xb.values()))
    XT = np.vstack(list(Xt.values()))
    rows = []
    for j, nm in enumerate(names):
        ks_b, _ = ks_p(Xm[:, j], XB[:, j])
        ks_t, _ = ks_p(Xm[:, j], XT[:, j])
        ks_bt, _ = ks_p(XB[:, j], XT[:, j])
        ref = [ks_p(Xb[m][:, j], np.vstack([Xb[o] for o in MATCHES if o != m])[:, j])[0] for m in MATCHES]
        _, p_thin = ks_p(Xm[::THIN, j], XB[::THIN, j])
        a, b = Xm[::THIN, j], XB[::THIN, j]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        try:
            ad = stats.anderson_ksamp([a, b]).pvalue if len(a) > 5 and len(b) > 5 else np.nan
        except Exception:  # noqa: BLE001
            ad = np.nan
        rows.append(dict(feature=nm, ks_match_bench=ks_b, ks_match_train=ks_t, ks_bench_train=ks_bt,
                         ks_ref_median=np.nanmedian(ref), ks_ref_max=np.nanmax(ref),
                         ratio_to_ref_max=ks_b / np.nanmax(ref) if np.nanmax(ref) > 0 else np.nan,
                         p_ks_thinned=p_thin, p_ad_thinned=ad,
                         nan_match=np.isnan(Xm[:, j]).mean(), nan_bench=np.isnan(XB[:, j]).mean(),
                         mean_match=np.nanmean(Xm[:, j]), mean_bench=np.nanmean(XB[:, j]),
                         sd_bench=np.nanstd(XB[:, j])))
    df = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    ok = df.p_ks_thinned.notna()
    df.loc[ok, "q_ks_thinned"] = multipletests(df.p_ks_thinned[ok], method="fdr_bh")[1]
    df["smd_match_bench"] = (df.mean_match - df.mean_bench) / df.sd_bench.replace(0, np.nan)
    fc = pd.read_csv(os.path.join(STATS, "feature_clusters.csv"))[["feature", "tree_importance", "auc_dead", "auc_ha"]]
    df = df.merge(fc, on="feature", how="left")
    df.to_csv(os.path.join(STATS, "shift_features.csv"), index=False)
    pd.set_option("display.width", 250)
    print("rows: match {}, bench {}, train {}".format(len(Xm), len(XB), len(XT)))
    print("features with KS(match, bench) > max DFL game-vs-rest KS: {} / {}".format(
        int((df.ratio_to_ref_max > 1).sum()), len(df)))
    print("features significant after thinning + BH (q < 0.05): {}".format(int((df.q_ks_thinned < 0.05).sum())))
    print("median KS match-bench {:.3f}, match-train {:.3f}, bench-train {:.3f}, DFL ref median {:.3f}".format(
        df.ks_match_bench.median(), df.ks_match_train.median(), df.ks_bench_train.median(), df.ks_ref_median.median()))
    cols = ["feature", "ks_match_bench", "ks_match_train", "ks_ref_max", "ratio_to_ref_max", "q_ks_thinned",
            "p_ad_thinned", "smd_match_bench", "nan_match", "nan_bench", "tree_importance", "auc_dead", "auc_ha"]
    print(df.sort_values("ks_match_bench", ascending=False).head(25)[cols].round(3).to_string())
    imp = df.sort_values("tree_importance", ascending=False).head(12)
    print("most important tree features and their shift:")
    print(imp[cols].round(3).to_string())
    top = df.assign(maxauc=df[["auc_dead", "auc_ha"]].max(1)).sort_values("maxauc", ascending=False).head(30)
    print("of the 30 most discriminative features, shifted beyond DFL game range: {}".format(
        int((top.ratio_to_ref_max > 1).sum())))
    return Xm, Xb, names, df


def part_trackstats():
    rows = []
    players, slot_team, ball, sign_half1, blocks, pitch, fps = match_raw()
    r = match_stats.stats(players, slot_team, ball, blocks, fps)
    r.update(source="match")
    rows.append(r)
    for m in MATCHES:
        for seed in SEEDS:
            slots, st, b, gt, bl, sg = bench.dfl2_prepared(m, seed)
            r = match_stats.stats(slots, st, b, bl, 25)
            r.update(source="bench", game=m, seed=seed)
            rows.append(r)
        z = np.load(os.path.join(WORK, "ds_v2/raw/{}_41.npz".format(m)), allow_pickle=True)
        bl = [tuple(int(v) for v in x) for x in z["blocks"]]
        r = match_stats.stats(z["slots"], z["slot_team"], z["ball"], bl, 25)
        r.update(source="train", game=m, seed=41)
        rows.append(r)
        print("trackstats", m, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS, "shift_trackstats_raw.csv"), index=False)
    keys = [k for k in df.columns if k not in ("source", "game", "seed", "graded_min")]
    out = []
    for k in keys:
        mv = df.loc[df.source == "match", k].iloc[0]
        for src in ("bench", "train"):
            v = df.loc[df.source == src, k].astype(float)
            out.append(dict(stat=k, ref=src, match=mv, ref_mean=v.mean(), ref_min=v.min(), ref_max=v.max(),
                            z=(mv - v.mean()) / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan,
                            inside_range=bool(v.min() <= mv <= v.max())))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(STATS, "shift_trackstats.csv"), index=False)
    print(out.round(3).to_string())


def part_c2st(Xm, Xb, names):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    chunk = 1500          # rows of 0.2 s = 5 min
    ball_cols = [j for j, n in enumerate(names) if n.startswith("ball_") or "ball" in n]
    sets = {"all features": list(range(len(names))),
            "without ball-derived features": [j for j in range(len(names)) if j not in ball_cols]}

    def c2st(A, B, cols, seed=0):
        sub_b = np.random.default_rng(seed).choice(len(B), min(len(B), 3 * len(A)), replace=False)
        B = B[np.sort(sub_b)]
        X = np.vstack([A[:, cols], B[:, cols]])
        y = np.r_[np.ones(len(A)), np.zeros(len(B))]
        grp = np.r_[np.arange(len(A)) // chunk, 10000 + np.arange(len(B)) // chunk]
        pr = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                                 random_state=0)
            clf.fit(X[tr], y[tr])
            pr[te] = clf.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, pr), pr, y

    rows = []
    XB = np.vstack([Xb[m] for m in MATCHES])
    for sname, cols in sets.items():
        a, _, _ = c2st(Xm, XB, cols)
        rows.append(dict(features=sname, comparison="match vs bench (7 games)", auc=a))
        for m in MATCHES:
            rest = np.vstack([Xb[o] for o in MATCHES if o != m])
            a, _, _ = c2st(Xb[m], rest, cols)
            rows.append(dict(features=sname, comparison="DFL {} vs other 6".format(SHORT[m]), auc=a))
        print("c2st", sname, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS, "shift_c2st.csv"), index=False)
    print(df.round(3).to_string())
    # which features drive the match-vs-bench classifier: drop-one-cluster is too costly; use the
    # univariate KS ranking above and the classifier's permutation importance on a holdout split
    from sklearn.inspection import permutation_importance
    sub = np.random.default_rng(1).choice(len(XB), 3 * len(Xm), replace=False)
    X = np.vstack([Xm, XB[sub]])
    y = np.r_[np.ones(len(Xm)), np.zeros(len(sub))]
    half = np.r_[np.arange(len(Xm)) < len(Xm) // 2, np.arange(len(sub)) % 2 == 0]
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[half], y[half])
    pi = permutation_importance(clf, X[~half][::4], y[~half][::4], scoring="roc_auc", n_repeats=3, random_state=0)
    imp = pd.DataFrame(dict(feature=names, perm_importance=pi.importances_mean)).sort_values(
        "perm_importance", ascending=False)
    imp.to_csv(os.path.join(STATS, "shift_c2st_importance.csv"), index=False)
    print(imp.head(12).round(4).to_string())


def runs_stats(codes, blocks):
    runs = [(int(a), int(b) + 1) for _, a, b in blocks]
    g = graded_mask(len(codes), blocks)
    seg = [(a, b, c) for a, b, c in pm.segments(codes, runs) if c in (pm.H, pm.A)]
    mins = g.sum() / 25 / 60
    return dict(dead_share=float((codes[g] == pm.D).mean()), poss_per90=len(seg) * 90 / mins,
                mean_poss_s=float(np.mean([b - a for a, b, _ in seg]) / 25),
                short_lt2s=float(np.mean([(b - a) < 50 for a, b, _ in seg])))


def part_outputs():
    _, _, _, _, mblocks, _, _ = match_raw()
    rows = []
    sub = {"ensemble_vote": "ensemble_vote", "selftrain_ctrl": "selftrain_ctrl", "decision_tree": "decision_tree",
           "temporal_tcn": "temporal_tcn", "handover_mstcn": "handover_mstcn"}
    mcodes = {}
    for c in sub:
        p = os.path.join(CAND, c, "submission.csv")
        lab = pd.read_csv(p).label.values
        codes = np.array([pm.LABELS.index(v) for v in lab], np.int8)
        mcodes[c] = codes
        r = runs_stats(codes, mblocks)
        r.update(cand=c, data="match (prediction)")
        rows.append(r)
    mcodes["ensemble_ctrl"] = control.fold(mcodes["ensemble_vote"], 2.0,
                                           [(int(a), int(b) + 1) for _, a, b in mblocks])
    r = runs_stats(mcodes["ensemble_ctrl"], mblocks)
    r.update(cand="ensemble_ctrl", data="match (prediction)")
    rows.append(r)
    dis_rows = []
    sc = pd.read_csv(os.path.join(STATS, "scores.csv"))
    ctrl = sc[sc["T"].isin([1, 2, 3])].groupby(["cand", "game", "seed"]).spellF1.mean()
    for m in MATCHES:
        for seed in SEEDS:
            slots, st, ball, gt, blocks, sign = bench.dfl2_prepared(m, seed)
            for c in ["ensemble_vote", "ensemble_ctrl", "decision_tree", "temporal_tcn", "handover_mstcn"]:
                codes = np.load(os.path.join(PROBS, c, "dfl2_{}_{}_codes.npy".format(m, seed)))
                r = runs_stats(codes, blocks)
                r.update(cand=c, data="bench (prediction)", game=m, seed=seed)
                rows.append(r)
            for T in (0, 2):
                r = runs_stats(bench.controlled_labels(gt, T), blocks)
                r.update(cand="truth T={}".format(T), data="bench (truth)", game=m, seed=seed)
                rows.append(r)
            a = np.load(os.path.join(PROBS, "temporal_tcn", "dfl2_{}_{}_codes.npy".format(m, seed)))
            b = np.load(os.path.join(PROBS, "handover_mstcn", "dfl2_{}_{}_codes.npy".format(m, seed)))
            g = graded_mask(len(a), blocks)
            dis_rows.append(dict(game=m, seed=seed, disagree=float((a[g] != b[g]).mean()),
                                 ctrl_ensemble=ctrl[("ensemble_ctrl", m, seed)],
                                 ctrl_selftrain=ctrl[("selftrain_ctrl", m, seed)]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS, "shift_outputs.csv"), index=False)
    summ = df.groupby(["data", "cand"])[["dead_share", "poss_per90", "mean_poss_s", "short_lt2s"]].agg(["mean", "min", "max"])
    print(summ.round(3).to_string())
    dd = pd.DataFrame(dis_rows)
    g = mcodes["temporal_tcn"] != pm.X
    gm = graded_mask(len(mcodes["temporal_tcn"]), mblocks)
    match_dis = float((mcodes["temporal_tcn"][gm] != mcodes["handover_mstcn"][gm]).mean())
    import statsmodels.api as sm
    fit = sm.OLS(dd.ctrl_ensemble, sm.add_constant(dd.disagree)).fit(
        cov_type="cluster", cov_kwds=dict(groups=pd.factorize(dd.game)[0]))
    pred = fit.get_prediction(np.array([[1.0, match_dis]])).summary_frame(alpha=0.05)
    r_game = stats.pearsonr(dd.groupby("game").disagree.mean(), dd.groupby("game").ctrl_ensemble.mean())
    print("disagreement temporal_tcn vs handover_mstcn: bench {:.4f} [{:.4f}, {:.4f}], match {:.4f}".format(
        dd.disagree.mean(), dd.disagree.min(), dd.disagree.max(), match_dis))
    print("ctrl_mean(ensemble_ctrl) ~ disagreement: slope {:.3f} (game-clustered SE {:.3f}), game-level r = {:.2f} (p {:.3f})".format(
        fit.params.iloc[1], fit.bse.iloc[1], r_game[0], r_game[1]))
    print("label-free extrapolation to the match: {:.3f} (95% mean CI {:.3f}-{:.3f})".format(
        pred["mean"].iloc[0], pred["mean_ci_lower"].iloc[0], pred["mean_ci_upper"].iloc[0]))
    dd.to_csv(os.path.join(STATS, "shift_disagreement.csv"), index=False)
    with open(os.path.join(STATS, "shift_disagreement.txt"), "w") as f:
        f.write("match_dis {:.4f}\nslope {:.4f} se {:.4f}\nr_game {:.3f} p {:.4f}\npred {:.4f} [{:.4f}, {:.4f}]\n".format(
            match_dis, fit.params.iloc[1], fit.bse.iloc[1], r_game[0], r_game[1], pred["mean"].iloc[0],
            pred["mean_ci_lower"].iloc[0], pred["mean_ci_upper"].iloc[0]))
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(dd.disagree, dd.ctrl_ensemble, color=C1, s=18, label="bench v2 game-seeds")
    ax.axvline(match_dis, color=C2, lw=2, label="BAR-ATH disagreement")
    ax.set_xlabel("frame disagreement, temporal_tcn vs handover_mstcn")
    ax.set_ylabel("ensemble_ctrl ctrl_mean")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Member disagreement as a label-free error proxy", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "disagreement.png"), dpi=130)


import control  # noqa: E402


def plot_ks(df):
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    d = df.sort_values("ks_match_bench", ascending=False).head(20).iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y, d.ks_match_bench, color=C1, height=0.6, label="BAR-ATH vs bench v2")
    ax.scatter(d.ks_ref_max, y, color=C2, s=22, zorder=3, label="largest DFL game-vs-rest KS")
    ax.scatter(d.ks_match_train, y, color=C3, s=16, marker="s", zorder=3, label="BAR-ATH vs training rows")
    ax.set_yticks(y)
    ax.set_yticklabels(d.feature, fontsize=7)
    ax.set_xlabel("Kolmogorov-Smirnov distance")
    ax.set_title("Most shifted LDE features", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shift_ks.png"), dpi=130)


if __name__ == "__main__":
    Xm, Xb, names, df = part_features()
    plot_ks(df)
    part_c2st(Xm, Xb, names)
    part_outputs()
    part_trackstats()
