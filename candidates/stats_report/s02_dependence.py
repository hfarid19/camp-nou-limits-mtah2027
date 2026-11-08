"""Analysis 1: dependence structure (autocorrelation, effective sample size, design effect, ICC).

- Integrated autocorrelation time tau (Sokal automatic window, c = 5) of label indicators, of
  key features (ds_v2, stride 5 = 0.2 s) and of the per-frame error of ensemble_ctrl on bench v2
  (25 fps, against controlled labels at T = 2 s). ESS = n / tau; for a mean, design effect = tau.
- Variance components of the bench score: game vs seed-within-game, for ctrl_mean levels and for
  paired differences between candidates (one-way random-effects ANOVA, ICC).
Outputs: $STATS/dependence_tau.csv, $STATS/variance_components.csv, fig/acf.png
"""
import os

import numpy as np
import pandas as pd

from common import STATS, WORK, FIG, MATCHES, SEEDS, PROBS, bench, pm, plt_setup, C1, C2, C3, C4, GREY


def acf(x, maxlag):
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    x = np.where(ok, x - np.nanmean(x), 0.0)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[:maxlag + 1]
    fo = np.fft.rfft(ok.astype(float), 2 * n)
    cnt = np.fft.irfft(fo * np.conj(fo))[:maxlag + 1]
    r = r / np.maximum(cnt, 1)
    return r / r[0] if r[0] > 0 else r


def tau_int(rho, c=5.0):
    """Sokal: smallest window M with M >= c * tau(M)."""
    t = 1.0
    for M in range(1, len(rho)):
        t = 1.0 + 2.0 * rho[1:M + 1].sum()
        if M >= c * t:
            return t, M
    return t, len(rho) - 1


def main():
    rows, acfs = [], {}
    feats = ["ball_seen", "mean_speed", "diff_d1", "lastctrl", "ball_speed", "dmin", "H_since_ctrl"]
    for m in MATCHES:
        z = np.load(os.path.join(WORK, "ds_v2/ds/{}_41.npz".format(m)), allow_pickle=True)
        names = list(z["names"])
        X, y = z["X"], z["y"]
        idx = z["idx"]
        # stride 5 rows are contiguous within a half; break at jumps
        ser = {"label: dead (D vs alive)": (y == 2).astype(float),
               "label: home (H vs A, alive)": np.where(y == 2, np.nan, (y == 0).astype(float))}
        for f in feats:
            ser["feature: " + f] = X[:, names.index(f)]
        for k, v in ser.items():
            rho = acf(v, 3000)
            t, M = tau_int(rho)
            rows.append(dict(game=m, series=k, source="ds_v2 seed 41 (0.2 s rows)", n=int((~np.isnan(v)).sum()),
                             tau_rows=t, tau_s=t * 0.2, ess=(~np.isnan(v)).sum() / t))
            acfs.setdefault(k, []).append(rho[:1501])
        # per-frame error of ensemble_ctrl on bench v2 (25 fps)
        gt = bench.dfl2_prepared(m, 2027)[3]
        g2 = bench.controlled_labels(gt, 2)
        pred = np.load(os.path.join(PROBS, "ensemble_ctrl", "dfl2_{}_2027_codes.npy".format(m)))
        ok = g2 != pm.X
        err = np.where(ok, (pred != g2).astype(float), np.nan)
        rho = acf(err, 25 * 600)
        t, M = tau_int(rho)
        rows.append(dict(game=m, series="error of ensemble_ctrl (T=2)", source="bench v2 seed 2027 (25 fps)",
                         n=int(ok.sum()), tau_rows=t, tau_s=t / 25, ess=ok.sum() / t, err_rate=np.nanmean(err)))
        acfs.setdefault("error of ensemble_ctrl (T=2) [25 fps]", []).append(rho[:7501])
        # possession-level ICC of the error (design effect for frames nested in possessions)
        runs = pm.valid_runs(g2)
        seg_id = np.full(len(g2), -1)
        for i, (a, b, c) in enumerate(pm.segments(g2, runs)):
            seg_id[a:b] = i
        d = pd.DataFrame(dict(seg=seg_id[ok], e=err[ok]))
        grp = d.groupby("seg").e
        k = grp.size()
        msb = (k * (grp.mean() - d.e.mean()) ** 2).sum() / (len(k) - 1)
        msw = ((d.e - grp.transform("mean")) ** 2).sum() / (len(d) - len(k))
        k0 = (len(d) - (k ** 2).sum() / len(d)) / (len(k) - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        rows.append(dict(game=m, series="segment ICC of error", source="bench v2 seed 2027",
                         n=len(d), tau_rows=np.nan, tau_s=np.nan, ess=len(d) / (1 + (k0 - 1) * icc),
                         icc=icc, mean_seg_frames=k0, n_segments=len(k)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS, "dependence_tau.csv"), index=False)
    summ = df.groupby("series").agg(n=("n", "median"), tau_s=("tau_s", "median"), tau_rows=("tau_rows", "median"),
                                    ess=("ess", "median"), icc=("icc", "median"))
    summ["design_effect"] = summ.n / summ.ess
    print(summ.round(3).to_string())

    # variance components of the bench score (game vs seed)
    sc = pd.read_csv(os.path.join(STATS, "scores.csv"))
    ctrl = sc[sc["T"].isin([1, 2, 3])].groupby(["cand", "game", "seed"]).spellF1.mean().unstack("cand")
    vc = []

    def comp(v, label):
        g = v.groupby(level="game")
        a = len(g)
        nseed = 3
        msb = nseed * ((g.mean() - v.mean()) ** 2).sum() / (a - 1)
        msw = ((v - g.transform("mean")) ** 2).sum() / (a * (nseed - 1))
        s2g = max(0.0, (msb - msw) / nseed)
        vc.append(dict(quantity=label, mean=v.mean(), sd_game=np.sqrt(s2g), sd_seed=np.sqrt(msw),
                       icc_game=s2g / (s2g + msw) if s2g + msw > 0 else np.nan,
                       se_game_level=np.sqrt(msb / (a * nseed))))
    for c in ["ensemble_ctrl", "ensemble_vote", "selftrain_ctrl", "decision_tree", "learned_dfl_evidence+fold2"]:
        comp(ctrl[c], "level: " + c)
    for a, b in [("selftrain_ctrl", "ensemble_ctrl"), ("ensemble_ctrl", "ensemble_vote"),
                 ("decision_tree", "learned_dfl_evidence+fold2"), ("ensemble_vote", "handover_mstcn"),
                 ("ensemble_ctrl", "temporal_tcn+fold2")]:
        comp(ctrl[a] - ctrl[b], "diff: {} - {}".format(a, b))
    vc = pd.DataFrame(vc)
    vc.to_csv(os.path.join(STATS, "variance_components.csv"), index=False)
    print(vc.round(4).to_string())

    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    lag_s = np.arange(1501) * 0.2
    for (k, col) in zip(["label: dead (D vs alive)", "label: home (H vs A, alive)", "feature: diff_d1",
                         "feature: mean_speed"], [C1, C2, C3, C4]):
        axes[0].plot(lag_s, np.mean(acfs[k], 0), color=col, label=k)
    axes[0].set_xlim(0, 120)
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].set_xlabel("lag (s)")
    axes[0].set_ylabel("autocorrelation (mean of 7 games)")
    axes[0].set_title("Labels and features (training rows)", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    e = np.mean(acfs["error of ensemble_ctrl (T=2) [25 fps]"], 0)
    axes[1].plot(np.arange(len(e)) / 25, e, color=C1)
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_xlim(0, 60)
    axes[1].set_xlabel("lag (s)")
    axes[1].set_title("Per-frame error of ensemble_ctrl (bench v2)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "acf.png"), dpi=130)


if __name__ == "__main__":
    main()
