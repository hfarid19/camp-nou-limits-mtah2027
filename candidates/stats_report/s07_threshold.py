"""Analysis 6: sensitivity to the control threshold T (labels) and to the fold threshold (output).

- Candidate ranking against labels folded at T = 0 ... 4 s: Kendall tau between each T's ranking
  and the ctrl_mean ranking, and the game-bootstrap probability that each candidate ranks first.
- ensemble_vote + fold t (t = 1 ... 3 s) scored at every label T: the best t per T, regret of
  t = 2.0, paired game-level tests 2.0 vs 1.5 / 2.5 per label T, and a nested leave-one-game-out
  choice of t (picked on 6 games, scored on the 7th) to measure selection optimism.
Outputs: $STATS/rank_by_T.csv, $STATS/fold_grid.csv, fig/rank_by_T.png, fig/fold_grid.png
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import STATS, FIG, MATCHES, plt_setup, C1, C2, C3, C4, GREY, INK2


def main():
    sc = pd.read_csv(os.path.join(STATS, "scores.csv"))
    sc = sc[~sc.cand.isin(["ball_radar", "semimarkov_lattice", "ensemble_vote+fold2", "decision_tree_raw+fold2"])]
    gm = sc.groupby(["cand", "T", "game"]).spellF1.mean().reset_index()
    tab = gm.groupby(["cand", "T"]).spellF1.mean().unstack("T")
    tab["ctrl_mean"] = tab[[1, 2, 3]].mean(1)
    rk = tab.rank(ascending=False)
    rows = []
    for T in [c for c in tab.columns if c != "ctrl_mean"]:
        tau = stats.kendalltau(tab[T], tab["ctrl_mean"])
        rows.append(dict(T=T, kendall_tau_vs_ctrl_mean=tau.statistic, p=tau.pvalue,
                         best=tab[T].idxmax(), best_score=tab[T].max(),
                         rank_selftrain=rk.loc["selftrain_ctrl", T], rank_ensemble_ctrl=rk.loc["ensemble_ctrl", T],
                         rank_ensemble_vote=rk.loc["ensemble_vote", T]))
    rt = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(tab.sort_values("ctrl_mean", ascending=False).round(4).to_string())
    print(rt.round(3).to_string())
    # bootstrap P(rank 1) over games
    rng = np.random.default_rng(0)
    wide = gm.pivot_table(index=["cand", "T"], columns="game", values="spellF1")
    first = {T: {} for T in [0, 1, 2, 3, 4, "ctrl_mean"]}
    B = 2000
    for _ in range(B):
        pick = rng.choice(MATCHES, 7)
        mb = wide[pick].mean(1).unstack("T")
        mb["ctrl_mean"] = mb[[1, 2, 3]].mean(1)
        for T in first:
            w = mb[T].idxmax()
            first[T][w] = first[T].get(w, 0) + 1 / B
    for T, d in first.items():
        print("P(rank 1) at T={}: {}".format(T, {k: round(v, 3) for k, v in sorted(d.items(), key=lambda x: -x[1])}))
    rt.to_csv(os.path.join(STATS, "rank_by_T.csv"), index=False)

    # fold grid
    folds = ["ensemble_vote"] + ["ensemble_vote+fold{:g}".format(t) for t in (1.0, 1.5, 2.0, 2.5, 3.0)]
    sc2 = pd.read_csv(os.path.join(STATS, "scores.csv"))
    sc2.loc[sc2.cand == "ensemble_vote+fold2", "cand"] = "ensemble_vote+fold2"
    fg = sc2[sc2.cand.isin(folds)].groupby(["cand", "T", "game"]).spellF1.mean()
    grid = fg.groupby(["cand", "T"]).mean().unstack("T").loc[folds]
    grid["ctrl_mean"] = grid[[1, 2, 3]].mean(1)
    print(grid.round(4).to_string())
    out = []
    for T in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]:
        best = grid[T].idxmax()
        d2 = fg.loc["ensemble_vote+fold2", T]
        r = dict(T=T, best_fold=best, best=grid.loc[best, T], fold2=grid.loc["ensemble_vote+fold2", T],
                 regret_fold2=grid.loc[best, T] - grid.loc["ensemble_vote+fold2", T])
        for alt in ("ensemble_vote+fold1.5", "ensemble_vote+fold2.5"):
            d = (d2 - fg.loc[alt, T]).values
            r["diff_vs_" + alt[-7:]] = d.mean()
            r["p_vs_" + alt[-7:]] = stats.ttest_1samp(d, 0).pvalue
        out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(STATS, "fold_grid.csv"), index=False)
    print(out.round(4).to_string())
    # nested LOGO choice of the fold threshold on ctrl_mean
    cm = sc2[sc2.cand.isin(folds[1:]) & sc2["T"].isin([1, 2, 3])].groupby(["cand", "game"]).spellF1.mean().unstack("game")
    nested, picks = [], []
    for g in MATCHES:
        pick = cm.drop(columns=g).mean(1).idxmax()
        picks.append(pick)
        nested.append(cm.loc[pick, g])
    print("nested LOGO fold choice: picks {}, nested ctrl_mean {:.4f} vs in-sample best {:.4f} (optimism {:+.4f})".format(
        sorted(set(picks)), np.mean(nested), cm.mean(1).max(), cm.mean(1).max() - np.mean(nested)))

    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    Ts = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
    show = {"selftrain_ctrl": C1, "ensemble_ctrl": C2, "ensemble_vote": C3, "handover_mstcn+fold2": C4,
            "learned_dfl_evidence+fold2": "#6250d6", "decision_tree": "#e87ba4"}
    for c, col in show.items():
        ax.plot(Ts, [tab.loc[c, T] for T in Ts], marker="o", ms=4, color=col, label=c)
    others = [c for c in tab.index if c not in show and c != "v1_original"]
    for c in others:
        ax.plot(Ts, [tab.loc[c, T] for T in Ts], color=GREY, lw=0.8, alpha=0.6)
    ax.set_xlabel("label control threshold T (s); grey = other candidates")
    ax.set_ylabel("spellF1 (mean of 7 games)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("Scores against labels folded at T", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rank_by_T.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cols = [C1, C2, C3, C4, "#6250d6", "#e87ba4"]
    for (c, col) in zip(folds, cols):
        ax.plot(Ts, [grid.loc[c, T] for T in Ts], marker="o", ms=4, color=col,
                label="no fold" if c == "ensemble_vote" else "fold " + c.split("fold")[1] + " s")
    ax.set_xlabel("label control threshold T (s)")
    ax.set_ylabel("spellF1 of ensemble_vote output")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fold threshold on the output vs label T", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fold_grid.png"), dpi=130)


if __name__ == "__main__":
    main()
