"""Analysis 4: univariate class separation of the 194 LDE features, and redundancy.

Data: ds_v2 training rows (seed 41 only; the 3 seeds share labels and differ only in degradation),
every stride-5 row (0.2 s), 7 games. Two tasks: D vs alive (all rows) and H vs A (alive rows).
Per feature and task:
  - pooled AUC (rank-based, NaN rows dropped) and standardised mean difference (SMD, pooled sd)
  - the same per game; game-level p-value = one-sample t (df 6) of the 7 per-game SMDs against 0
    (frames are not independent, games are), and sign consistency (games with the pooled sign)
  - Benjamini-Hochberg FDR over the 2 x 194 tests
  - missing share (NaN) by class: NaN-ness is itself evidence (ball not seen -> dead)
Redundancy: Spearman correlations on 30k pooled rows (pairwise complete), hierarchical clusters
at |rho| >= 0.9 / 0.95, VIF from the inverse correlation matrix (median-imputed), and the
impurity importance of the shipped decision tree (tree.npz) as a model-based relevance check.
Outputs: $STATS/features_univariate.csv, $STATS/feature_clusters.csv, fig/feature_auc.png
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from common import STATS, WORK, FIG, CAND, MATCHES, SHORT, plt_setup, C1, C2, C3, GREY, INK2

warnings.filterwarnings("ignore")


def auc(x, y):
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(x)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def smd(x, y):
    a, b = x[y & ~np.isnan(x)], x[~y & ~np.isnan(x)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def main():
    Xs, ys, gs = [], [], []
    for gi, m in enumerate(MATCHES):
        z = np.load(os.path.join(WORK, "ds_v2/ds/{}_41.npz".format(m)), allow_pickle=True)
        names, kinds = list(z["names"]), list(z["kinds"])
        Xs.append(z["X"].astype(np.float64))
        ys.append(z["y"])
        gs.append(np.full(len(z["y"]), gi))
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
    tasks = {"D vs alive": (np.ones(len(y), bool), y == 2), "H vs A": (y < 2, y == 0)}
    rows = []
    for t, (sub, pos) in tasks.items():
        for j, nm in enumerate(names):
            x = X[sub, j]
            p = pos[sub]
            gg = g[sub]
            pg_smd = np.array([smd(x[gg == i], p[gg == i]) for i in range(7)])
            pg_auc = np.array([auc(x[gg == i], p[gg == i]) for i in range(7)])
            sm_ = smd(x, p)
            ok = ~np.isnan(pg_smd)
            tp = stats.ttest_1samp(pg_smd[ok], 0).pvalue if ok.sum() >= 3 and np.std(pg_smd[ok]) > 0 else np.nan
            rows.append(dict(task=t, feature=nm, kind=kinds[j], auc=auc(x, p), smd=sm_,
                             auc_game_min=np.nanmin(np.where(np.nanmean(pg_auc) >= 0.5, pg_auc, 1 - pg_auc)),
                             smd_game_mean=np.nanmean(pg_smd), smd_game_sd=np.nanstd(pg_smd, ddof=1),
                             sign_consistent=int(np.sum(np.sign(pg_smd) == np.sign(sm_))),
                             p_game=tp, p_naive=stats.ttest_ind(x[p & ~np.isnan(x)], x[~p & ~np.isnan(x)]).pvalue,
                             nan_pos=np.isnan(x[p]).mean(), nan_neg=np.isnan(x[~p]).mean()))
    df = pd.DataFrame(rows)
    df["auc_sep"] = (df.auc - 0.5).abs() + 0.5
    from statsmodels.stats.multitest import multipletests
    okp = df.p_game.notna()
    df.loc[okp, "q_bh"] = multipletests(df.p_game[okp], method="fdr_bh")[1]
    df.loc[df.p_naive.notna(), "q_naive_bh"] = multipletests(df.p_naive.dropna(), method="fdr_bh")[1]

    # tree importance
    tz = np.load(os.path.join(CAND, "decision_tree", "tree.npz"), allow_pickle=True)
    cnt = tz["count"].sum(1).astype(float)
    imp = np.zeros(len(names))
    for i in range(len(tz["feature"])):
        if tz["left"][i] >= 0 and tz["feature"][i] >= 0:
            l, r = tz["left"][i], tz["right"][i]
            imp[tz["feature"][i]] += cnt[i] * tz["impurity"][i] - cnt[l] * tz["impurity"][l] - cnt[r] * tz["impurity"][r]
    imp = imp / imp.sum()
    df["tree_importance"] = df.feature.map(dict(zip(names, imp)))

    # redundancy
    rng = np.random.default_rng(0)
    pick = rng.choice(len(X), 30000, replace=False)
    S = pd.DataFrame(X[pick], columns=names)
    R = S.corr(method="spearman", min_periods=2000).fillna(0.0).values.copy()
    np.fill_diagonal(R, 1.0)
    D = 1 - np.abs(R)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    L = linkage(squareform(D, checks=False), "average")
    cl90 = fcluster(L, 0.10, "distance")
    cl95 = fcluster(L, 0.05, "distance")
    Z = S.fillna(S.median())
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)
    C = np.corrcoef(Z.values.T)
    C = np.nan_to_num(C)
    try:
        vif = np.diag(np.linalg.pinv(C))
    except np.linalg.LinAlgError:
        vif = np.full(len(names), np.nan)
    feat = pd.DataFrame(dict(feature=names, kind=kinds, cluster90=cl90, cluster95=cl95, vif=vif))
    best_dead = df[df.task == "D vs alive"].set_index("feature").auc_sep
    best_ha = df[df.task == "H vs A"].set_index("feature").auc_sep
    feat["auc_dead"] = feat.feature.map(best_dead)
    feat["auc_ha"] = feat.feature.map(best_ha)
    feat["tree_importance"] = imp
    feat["max_auc"] = feat[["auc_dead", "auc_ha"]].max(1)
    feat["cluster90_size"] = feat.groupby("cluster90").feature.transform("size")
    feat["rep90"] = feat.groupby("cluster90").max_auc.transform("max") == feat.max_auc
    feat.to_csv(os.path.join(STATS, "feature_clusters.csv"), index=False)
    df.to_csv(os.path.join(STATS, "features_univariate.csv"), index=False)

    pd.set_option("display.width", 250)
    for t in tasks:
        d = df[df.task == t].sort_values("auc_sep", ascending=False)
        print("== {}: top 15 by AUC".format(t))
        print(d.head(15)[["feature", "auc", "smd", "auc_game_min", "sign_consistent", "p_game", "q_bh",
                          "nan_pos", "nan_neg", "tree_importance"]].round(4).to_string())
        print("{}: significant at q<0.05 game-level: {} / {}; naive per-frame: {} / {}; AUC sep >= 0.6: {}".format(
            t, int((d.q_bh < 0.05).sum()), len(d), int((d.q_naive_bh < 0.05).sum()), len(d),
            int((d.auc_sep >= 0.6).sum())))
        if t == "H vs A":
            sym = d[d.kind == "sym"]
            print("sym features on H vs A: max |AUC-0.5| {:.3f}, any q<0.05: {}".format(
                (sym.auc - 0.5).abs().max(), int((sym.q_bh < 0.05).sum())))
    print("clusters at |rho|>=0.90: {} (from 194); at 0.95: {}".format(len(set(cl90)), len(set(cl95))))
    print("VIF > 10: {}, > 100: {}, > 1000: {}".format(int((vif > 10).sum()), int((vif > 100).sum()), int((vif > 1000).sum())))
    big = feat[feat.cluster90_size >= 4].sort_values(["cluster90", "max_auc"], ascending=[True, False])
    for c, grp in big.groupby("cluster90"):
        print("  cluster {} ({}): {}".format(c, len(grp), ", ".join(grp.feature.head(10))))
    weak = feat[(feat.max_auc < 0.55) & (feat.tree_importance < 0.001)]
    print("weak on both tasks (AUC sep < 0.55) and unused by tree (<0.1% importance): {}".format(len(weak)))
    print("  " + ", ".join(weak.feature))
    print("tree importance top 12:")
    print(feat.sort_values("tree_importance", ascending=False).head(12)[["feature", "tree_importance", "auc_dead", "auc_ha", "cluster90_size"]].round(4).to_string())

    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, t, col in zip(axes, tasks, [C1, C2]):
        d = df[df.task == t].sort_values("auc_sep", ascending=False).head(20).iloc[::-1]
        ax.barh(range(len(d)), d.auc_sep - 0.5, left=0.5, color=[col if q < 0.05 else GREY for q in d.q_bh],
                height=0.7)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.feature, fontsize=7)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("AUC (folded to >= 0.5); grey = not significant at game level after BH")
        ax.set_title("Top 20 features: " + t, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "feature_auc.png"), dpi=130)


if __name__ == "__main__":
    main()
