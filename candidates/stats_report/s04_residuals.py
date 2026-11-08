"""Analysis 3: heteroskedasticity, residuals, calibration, influential games.

(a) Bench score vs game characteristics. Most characteristics are per game (labels do not change
    with the degradation seed), so there are 7 independent points: per-predictor game-level
    Pearson / Spearman correlations (multiple regression with 4+ predictors on 7 points is not
    estimable in any useful sense, so it is skipped). Brown-Forsythe (median Levene) across games
    on the 3 seed values tests whether seed noise differs between games.
(b) Per-frame logistic models on the training rows (ds_v2, seed 41, every stride-5 row): dead vs
    alive, and home vs away on alive rows, with a few top features (standardised). Standard errors:
    naive iid, cluster-robust by possession spell, cluster-robust by game (CR1, 7 clusters, t(6)
    reference). Residual autocorrelation within games: Durbin-Watson and Ljung-Box on Pearson
    residuals. Influence: leave-one-game-out change of every coefficient.
(c) Calibration on bench v2 of the candidates that save real probabilities (every 5th frame):
    Brier and ECE (15 equal-width bins) per class against raw DFL labels (what they were trained
    for), with game-cluster bootstrap CIs, plus reliability curves.
Outputs: $STATS/game_char_corr.csv, $STATS/levene.csv, $STATS/logit_se.csv, $STATS/logit_resid.csv,
$STATS/logit_influence.csv, $STATS/calibration.csv, fig/reliability.png, fig/score_vs_games.png
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import STATS, WORK, FIG, MATCHES, SEEDS, PROBS, SHORT, bench, pm, plt_setup, C1, C2, C3, C4, GREY

warnings.filterwarnings("ignore")
CHARS = ["dead_share", "ball_missing", "poss_per90_raw", "mean_spell_s", "short_share_lt2s", "home_share"]


def part_a():
    sc = pd.read_csv(os.path.join(STATS, "scores.csv"))
    gch = pd.read_csv(os.path.join(STATS, "games.csv"))
    ctrl = sc[sc["T"].isin([1, 2, 3])].groupby(["cand", "game", "seed"]).spellF1.mean().unstack("cand")
    ys = {"ensemble_ctrl": ctrl["ensemble_ctrl"], "selftrain_ctrl": ctrl["selftrain_ctrl"],
          "gain of fold (ensemble_ctrl - ensemble_vote)": ctrl["ensemble_ctrl"] - ctrl["ensemble_vote"],
          "gain of self-training (selftrain - ensemble_ctrl)": ctrl["selftrain_ctrl"] - ctrl["ensemble_ctrl"],
          "tree - LDE+fold": ctrl["decision_tree"] - ctrl["learned_dfl_evidence+fold2"]}
    gg = gch.groupby("game")[CHARS].mean()
    rows = []
    for yn, y in ys.items():
        yg = y.groupby(level="game").mean()
        for c in CHARS:
            r, p = stats.pearsonr(gg[c], yg[gg.index])
            rs, ps = stats.spearmanr(gg[c], yg[gg.index])
            rows.append(dict(y=yn, x=c, pearson_r=r, p_pearson=p, spearman_rho=rs, p_spearman=ps))
    df = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    df["bh_pearson"] = multipletests(df.p_pearson, method="fdr_bh")[1]
    df.to_csv(os.path.join(STATS, "game_char_corr.csv"), index=False)
    print(gg.round(3).to_string())
    print(df.round(3).to_string())
    lev = []
    for yn, y in ys.items():
        groups = [y.xs(g, level="game").values for g in MATCHES]
        st, p = stats.levene(*groups, center="median")
        sds = [np.std(gr, ddof=1) for gr in groups]
        lev.append(dict(y=yn, brown_forsythe_W=st, p=p, seed_sd_min=min(sds), seed_sd_max=max(sds),
                        game_with_max_sd=SHORT[MATCHES[int(np.argmax(sds))]]))
    lev = pd.DataFrame(lev)
    lev.to_csv(os.path.join(STATS, "levene.csv"), index=False)
    print(lev.round(4).to_string())
    # Breusch-Pagan on the 21 game-seed rows for the one predictor that varies within game
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    d = gch.set_index(["game", "seed"])
    y = ctrl["ensemble_ctrl"].reindex(d.index)
    Xd = sm.add_constant(d[["dead_share", "ball_missing"]])
    fit = sm.OLS(y.values, Xd.values).fit(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(d.index.get_level_values(0))[0]))
    bp = het_breuschpagan(fit.resid, Xd.values)
    print("OLS ensemble_ctrl ~ dead_share + ball_missing (21 rows, game-clustered SE):")
    print(pd.DataFrame(dict(coef=fit.params, se=fit.bse, p=fit.pvalues), index=Xd.columns).round(4).to_string())
    print("Breusch-Pagan LM p = {:.3f}".format(bp[1]))
    with open(os.path.join(STATS, "bp.txt"), "w") as f:
        f.write("coef {}\nse {}\np {}\nBP_p {:.4f}\n".format(fit.params.round(4).tolist(), fit.bse.round(4).tolist(),
                                                             fit.pvalues.round(4).tolist(), bp[1]))
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    yg = ys["ensemble_ctrl"].groupby(level="game").mean()
    for ax, c in zip(axes, ["dead_share", "poss_per90_raw", "short_share_lt2s"]):
        ax.scatter(gg[c], yg[gg.index], color=C1, s=30, zorder=3)
        for g in gg.index:
            ax.annotate(SHORT[g], (gg.loc[g, c], yg[g]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        r = df[(df.y == "ensemble_ctrl") & (df.x == c)].iloc[0]
        ax.set_title("{}  r={:.2f} (p={:.2f})".format(c, r.pearson_r, r.p_pearson), loc="left", fontsize=8.5)
        ax.set_xlabel(c)
    axes[0].set_ylabel("ensemble_ctrl ctrl_mean (game)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "score_vs_games.png"), dpi=130)


def spell_ids(y):
    """id of the maximal same-label run each row belongs to (on the stride-5 grid)."""
    ch = np.concatenate(([True], y[1:] != y[:-1]))
    return np.cumsum(ch)


def part_b():
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
    specs = {"dead vs alive": (["ball_seen_m2.0", "mean_speed_m2.0", "stationary_m2.0", "ball_out_m2.0",
                                "any_since_ctrl", "ball_speed_m2.0"], None),
             "home vs away (alive rows)": (["lastctrl_m0.5", "diff_d1_m0.5", "diff_c5_m0.5", "diff_cx_m2.0",
                                            "diff_fwd_m2.0"], "alive")}
    Xs, ys, gs, ss = {k: [] for k in specs}, {k: [] for k in specs}, {k: [] for k in specs}, {k: [] for k in specs}
    off = 0
    for gi, m in enumerate(MATCHES):
        z = np.load(os.path.join(WORK, "ds_v2/ds/{}_41.npz".format(m)), allow_pickle=True)
        names = list(z["names"])
        X, y = z["X"].astype(np.float64), z["y"]
        sid = spell_ids(y) + off
        off = sid.max() + 1
        for k, (feats, sub) in specs.items():
            F = X[:, [names.index(f) for f in feats]]
            keep = (y < 2) if sub else np.ones(len(y), bool)
            target = (y == 0) if sub else (y == 2)
            Xs[k].append(F[keep])
            ys[k].append(target[keep].astype(float))
            gs[k].append(np.full(keep.sum(), gi))
            ss[k].append(sid[keep])
    se_rows, res_rows, inf_rows = [], [], []
    for k, (feats, _) in specs.items():
        X = np.vstack(Xs[k])
        y, g, s = np.concatenate(ys[k]), np.concatenate(gs[k]), np.concatenate(ss[k])
        ok = ~np.isnan(X).any(1)
        X, y, g, s = X[ok], y[ok], g[ok], s[ok]
        mu, sd = X.mean(0), X.std(0)
        Z = sm.add_constant((X - mu) / sd)
        model = sm.GLM(y, Z, family=sm.families.Binomial())
        f0 = model.fit()
        fs = model.fit(cov_type="cluster", cov_kwds=dict(groups=s))
        fg = model.fit(cov_type="cluster", cov_kwds=dict(groups=g, use_correction=True))
        for j, nm in enumerate(["const"] + feats):
            tg = fg.params[j] / fg.bse[j]
            se_rows.append(dict(model=k, term=nm, coef=f0.params[j], se_naive=f0.bse[j], se_spell=fs.bse[j],
                                se_game=fg.bse[j], inflation_spell=fs.bse[j] / f0.bse[j],
                                inflation_game=fg.bse[j] / f0.bse[j], z_naive=f0.params[j] / f0.bse[j],
                                p_game_t6=2 * stats.t.sf(abs(tg), 6), n=len(y), n_spells=len(np.unique(s))))
        # residual autocorrelation within games (Pearson residuals, time order kept)
        r = f0.resid_pearson
        for gi in range(7):
            rg = r[g == gi]
            lb = acorr_ljungbox(rg, lags=[5, 25, 50], return_df=True)
            res_rows.append(dict(model=k, game=SHORT[MATCHES[gi]], durbin_watson=durbin_watson(rg),
                                 lag1_r=np.corrcoef(rg[:-1], rg[1:])[0, 1],
                                 lag25_r=np.corrcoef(rg[:-25], rg[25:])[0, 1],
                                 lb_p_lag5=lb.lb_pvalue.iloc[0], lb_p_lag50=lb.lb_pvalue.iloc[2]))
        # leave-one-game-out influence
        for gi in range(7):
            keep = g != gi
            fl = sm.GLM(y[keep], Z[keep], family=sm.families.Binomial()).fit()
            for j, nm in enumerate(["const"] + feats):
                inf_rows.append(dict(model=k, dropped=SHORT[MATCHES[gi]], term=nm, coef_full=f0.params[j],
                                     coef_drop=fl.params[j], dfbeta_in_game_se=(f0.params[j] - fl.params[j]) / fg.bse[j]))
    se = pd.DataFrame(se_rows)
    se.to_csv(os.path.join(STATS, "logit_se.csv"), index=False)
    print(se.round(3).to_string())
    rr = pd.DataFrame(res_rows)
    rr.to_csv(os.path.join(STATS, "logit_resid.csv"), index=False)
    print(rr.round(3).to_string())
    inf = pd.DataFrame(inf_rows)
    inf.to_csv(os.path.join(STATS, "logit_influence.csv"), index=False)
    worst = inf.loc[inf.groupby(["model", "term"]).dfbeta_in_game_se.apply(lambda v: v.abs().idxmax())]
    print(worst.round(3).to_string())


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def part_c():
    cands = ["temporal_tcn", "handover_mstcn", "ball_radar", "pitch_control", "skillcorner_domain",
             "pathcrf_dynamic", "ballfree_dead", "semimarkov_lattice", "decision_tree"]
    data = {}
    for m in MATCHES:
        for seed in SEEDS:
            gt = bench.dfl2_prepared(m, seed)[3]
            idx = np.where(gt != pm.X)[0][::5]
            data[(m, seed)] = (idx, gt[idx])
    rows, curves = [], {}
    rng = np.random.default_rng(1)
    for c in cands:
        per_game = {m: [] for m in MATCHES}
        allp, ally = [], []
        for (m, seed), (idx, y) in data.items():
            P = np.load(os.path.join(PROBS, c, "dfl2_{}_{}.npy".format(m, seed)))[idx].astype(np.float64)
            P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
            Y = np.eye(3)[y]
            per_game[m].append((P, Y))
            allp.append(P)
            ally.append(Y)
        P, Y = np.vstack(allp), np.vstack(ally)
        curves[c] = (P, Y)
        for k, lab in enumerate("HAD"):
            br = np.mean((P[:, k] - Y[:, k]) ** 2)
            e = ece(P[:, k], Y[:, k])
            boots_b, boots_e = [], []
            for _ in range(300):
                pick = rng.choice(MATCHES, 7)
                Pb = np.vstack([q for g in pick for q, _ in per_game[g]])
                Yb = np.vstack([q for g in pick for _, q in per_game[g]])
                boots_b.append(np.mean((Pb[:, k] - Yb[:, k]) ** 2))
                boots_e.append(ece(Pb[:, k], Yb[:, k]))
            rows.append(dict(cand=c, cls=lab, brier=br, brier_lo=np.percentile(boots_b, 2.5),
                             brier_hi=np.percentile(boots_b, 97.5), ece=e, ece_lo=np.percentile(boots_e, 2.5),
                             ece_hi=np.percentile(boots_e, 97.5), mean_p=P[:, k].mean(), freq=Y[:, k].mean()))
        rows.append(dict(cand=c, cls="all", brier=np.mean(((P - Y) ** 2).sum(1)),
                         acc=float((P.argmax(1) == Y.argmax(1)).mean())))
    cal = pd.DataFrame(rows)
    cal.to_csv(os.path.join(STATS, "calibration.csv"), index=False)
    print(cal.round(4).to_string())
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    edges = np.linspace(0, 1, 16)
    for ax, k, lab in zip(axes, range(3), ["H (home)", "A (away)", "D (dead)"]):
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
        for c, col in zip(["temporal_tcn", "handover_mstcn", "decision_tree"], [C1, C2, C3]):
            P, Y = curves[c]
            b = np.clip(np.digitize(P[:, k], edges) - 1, 0, 14)
            xs = [P[b == i, k].mean() for i in range(15) if (b == i).sum() > 50]
            ys_ = [Y[b == i, k].mean() for i in range(15) if (b == i).sum() > 50]
            ax.plot(xs, ys_, marker="o", ms=4, color=col, label=c)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency (raw DFL labels)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "reliability.png"), dpi=130)


if __name__ == "__main__":
    print("== (a) game characteristics ==")
    part_a()
    print("== (b) per-frame logistic models ==")
    part_b()
    print("== (c) calibration ==")
    part_c()
