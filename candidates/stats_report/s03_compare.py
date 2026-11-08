"""Analysis 2: are the candidates' ctrl_mean differences real?

Unit of inference = game (7). Seeds (3 per game) are nested inside games and averaged first for the
game-level tests. Per comparison (a - b, paired on the same game and seed):
  - mean difference and 95% CI from the game-level paired t (df = 6), effect size dz = mean / sd
  - paired t, exact Wilcoxon signed-rank and exact sign-flip permutation (2^7 = 128) on game means
  - game-cluster bootstrap 95% percentile CI (resample games with their 3 seeds, 20k draws)
  - linear mixed model diff ~ 1 + (1 | game) (REML, statsmodels MixedLM) and the LRT for the game
    variance (boundary: p halved)
  - naive t over the 21 game-seeds (shown only to say how wrong it is when the game ICC is high)
  - minimum detectable difference (80% power, two-sided 5%) with 7 games, from the game-level sd
Holm is applied (a) within the key-comparison family and (b) within "every candidate vs the
best" family, on the sign-flip permutation p (exact, assumption-light).
Outputs: $STATS/compare_key.csv, $STATS/compare_all.csv, fig/paired_diffs.png
"""
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import STATS, FIG, SHORT, plt_setup, C1, C2, C3, GREY, INK2

warnings.filterwarnings("ignore")


def ctrl_table():
    sc = pd.read_csv(os.path.join(STATS, "scores.csv"))
    return sc[sc["T"].isin([1, 2, 3])].groupby(["cand", "game", "seed"]).spellF1.mean().unstack("cand")


def signflip_p(d):
    d = np.asarray(d)
    n = len(d)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    null = np.abs((signs * d).mean(1))
    return float((null >= obs - 1e-12).mean())


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def compare(t, a, b, rng):
    d = (t[a] - t[b]).rename("d").reset_index()
    gm = d.groupby("game").d.mean().values
    n = len(gm)
    m, sd = gm.mean(), gm.std(ddof=1)
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    out = dict(a=a, b=b, diff=m, ci_lo=m - tc * se, ci_hi=m + tc * se, dz=m / sd if sd > 0 else np.inf,
               p_t=float(stats.ttest_1samp(gm, 0).pvalue),
               p_wilcoxon=float(stats.wilcoxon(gm, method="exact").pvalue) if np.any(gm != 0) else 1.0,
               p_signflip=signflip_p(gm), games_won=int((gm > 0).sum()),
               seeds_won=int((d.d > 1e-9).sum()), n_pairs=len(d))
    # cluster bootstrap over games
    games = d.game.unique()
    by = {g: d.d[d.game == g].values for g in games}
    boots = np.empty(20000)
    for i in range(len(boots)):
        pick = rng.choice(games, len(games))
        boots[i] = np.mean([by[g].mean() for g in pick])
    out["boot_lo"], out["boot_hi"] = np.percentile(boots, [2.5, 97.5])
    # mixed model
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm("d ~ 1", d, groups=d["game"])
        fit = md.fit(reml=True)
        out["lmm_est"], out["lmm_se"] = fit.params["Intercept"], fit.bse["Intercept"]
        out["lmm_p"] = float(fit.pvalues["Intercept"])
        out["lmm_var_game"], out["lmm_var_seed"] = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
        fml = md.fit(reml=False)
        f0 = smf.ols("d ~ 1", d).fit()
        lr = max(0.0, 2 * (fml.llf - f0.llf))
        out["lrt_game_p"] = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0
    except Exception as e:  # noqa: BLE001
        out["lmm_err"] = str(e)[:80]
    # naive
    out["naive21_se"] = d.d.std(ddof=1) / np.sqrt(len(d))
    out["naive21_p"] = float(stats.ttest_1samp(d.d, 0).pvalue)
    out["se_game"] = se
    out["mdd80"] = (tc + stats.t.ppf(0.80, n - 1)) * se
    # games needed to detect this diff at 80% power (normal approx on game-level sd)
    out["games_needed"] = int(np.ceil(((1.96 + 0.8416) * sd / abs(m)) ** 2)) if m != 0 else np.nan
    return out, gm


def main():
    rng = np.random.default_rng(0)
    t = ctrl_table()
    key = [("ensemble_ctrl", "ensemble_vote"), ("selftrain_ctrl", "ensemble_ctrl"),
           ("decision_tree", "learned_dfl_evidence+fold2"), ("ensemble_ctrl", "temporal_tcn+fold2"),
           ("ensemble_ctrl", "handover_mstcn+fold2"), ("ensemble_vote+fold2", "ensemble_vote+fold1.5"),
           ("ensemble_vote+fold2", "ensemble_vote+fold2.5")]
    res, gms = [], {}
    for a, b in key:
        r, gm = compare(t, a, b, rng)
        res.append(r)
        gms[(a, b)] = gm
    kd = pd.DataFrame(res)
    kd["holm_signflip"] = holm(kd.p_signflip)
    kd["holm_t"] = holm(kd.p_t)
    kd.to_csv(os.path.join(STATS, "compare_key.csv"), index=False)
    cols = ["a", "b", "diff", "ci_lo", "ci_hi", "boot_lo", "boot_hi", "dz", "p_t", "p_wilcoxon", "p_signflip",
            "holm_signflip", "holm_t", "lmm_est", "lmm_se", "lmm_p", "lrt_game_p", "naive21_p", "games_won",
            "seeds_won", "mdd80", "games_needed"]
    pd.set_option("display.width", 250)
    print(kd[cols].round(4).to_string())

    # every candidate vs the best (selftrain_ctrl) and vs ensemble_ctrl
    excl = {"ensemble_vote+fold2"}            # identical to ensemble_ctrl
    allr = []
    for ref in ("selftrain_ctrl", "ensemble_ctrl"):
        rr = []
        for c in t.columns:
            if c in excl or c == ref:
                continue
            r, _ = compare(t, ref, c, rng)
            rr.append(r)
        rr = pd.DataFrame(rr)
        rr["holm_signflip"] = holm(rr.p_signflip)
        rr["holm_t"] = holm(rr.p_t)
        allr.append(rr)
    ad = pd.concat(allr)
    ad.to_csv(os.path.join(STATS, "compare_all.csv"), index=False)
    print(ad[["a", "b", "diff", "ci_lo", "ci_hi", "p_t", "p_signflip", "holm_t", "holm_signflip", "games_won",
              "lmm_p"]].sort_values(["a", "diff"]).round(4).to_string())

    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    labels = []
    for i, ((a, b), gm) in enumerate(gms.items()):
        r = kd.iloc[i]
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=C1, lw=2, solid_capstyle="round")
        ax.scatter(gm, np.full(len(gm), i) + 0.18, s=14, color=GREY, zorder=3)
        ax.scatter([r["diff"]], [i], s=40, color=C1, zorder=4)
        labels.append("{}\n  vs {}".format(a, b))
        ax.text(r.ci_hi + 0.002, i - 0.1, ("Holm p<0.001" if r.holm_t < 0.001 else "Holm p={:.3f}".format(r.holm_t)), fontsize=7.5, color=INK2)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("difference in ctrl_mean (a - b); dot = mean, bar = 95% CI over 7 games, grey = per game")
    ax.set_title("Key paired comparisons on bench v2", loc="left")
    ax.set_xlim(right=0.055)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.14, top=0.92)
    fig.savefig(os.path.join(FIG, "paired_diffs.png"), dpi=130)


if __name__ == "__main__":
    main()
