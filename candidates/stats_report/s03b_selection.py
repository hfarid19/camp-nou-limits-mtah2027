"""Analysis 2b: how much of selftrain_ctrl's +0.007 survives its own variant selection?

selftrain_ctrl's shipped setting (LightGBM 100 rounds, min_dist 12, soft weight w = 1.5) was picked
among offline variants scored on the same 21 bench-v2 game-seeds. This rebuilds the grid
{7 saved P_st variants} x {w = 1.0, 1.5, 2.0, 3.0} with the offline harness (exp_combo.soft_add +
fold 2 s) and runs a nested leave-one-game-out selection: for each game, pick the variant with the
best mean ctrl_mean on the other 6 games, score it on the held-out game. The nested gain over
ensemble_ctrl is the honest estimate of what the procedure (not the lucky variant) delivers.
Output: $STATS/selftrain_grid.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

WORK = os.environ["MTAH_WORK"]
STATS = os.path.join(WORK, "stats")

OFF = os.path.join(WORK, "arch/selftrain_ctrl/offline")
sys.path.insert(0, OFF)
import common as offc  # noqa: E402  (the offline harness common.py, not stats_report/common.py)
import exp_combo  # noqa: E402

VARS = sorted(os.listdir(os.path.join(OFF, "pst")))
WS = [1.0, 1.5, 2.0, 3.0]


def main():
    path = os.path.join(STATS, "selftrain_grid.csv")
    if not os.path.isfile(path):
        rows = []
        for gi, m in enumerate(offc.MATCHES):
            for seed in offc.SEEDS:
                g = offc.load_game(gi, seed)
                base = offc.score(offc.fold(g["ev"], g), g)["ctrl"]
                rows.append(dict(game=m, seed=seed, variant="ensemble_ctrl", ctrl=base))
                for v in VARS:
                    P = exp_combo.load_P(v, g)
                    for w in WS:
                        c = offc.score(offc.fold(exp_combo.soft_add(g, P, w), g), g)["ctrl"]
                        rows.append(dict(game=m, seed=seed, variant="{}|w{}".format(v, w), ctrl=c))
                print(m, seed, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.read_csv(path)
    gm = df.groupby(["variant", "game"]).ctrl.mean().unstack("game")
    base = gm.loc["ensemble_ctrl"]
    gain = gm.drop("ensemble_ctrl") - base
    insample = gain.mean(1).sort_values(ascending=False)
    print("in-sample gain over ensemble_ctrl (top 8 of {}):".format(len(insample)))
    print(insample.head(8).round(4).to_string())
    print("shipped lgb100_m12|w1.5: {:+.4f}, rank {}".format(insample["lgb100_m12|w1.5"],
                                                          list(insample.index).index("lgb100_m12|w1.5") + 1))
    print("share of variants with positive mean gain: {:.2f}".format((insample > 0).mean()))
    nested = []
    for g in gain.columns:
        pick = gain.drop(columns=g).mean(1).idxmax()
        nested.append(gain.loc[pick, g])
        print("  held out {}: picked {:22s} gain {:+.4f}".format(g[-3:], pick, gain.loc[pick, g]))
    nested = np.array(nested)
    se = nested.std(ddof=1) / np.sqrt(7)
    tc = stats.t.ppf(0.975, 6)
    print("nested LOGO gain {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  p_t {:.4f}  games won {}/7".format(
        nested.mean(), nested.mean() - tc * se, nested.mean() + tc * se,
        stats.ttest_1samp(nested, 0).pvalue, int((nested > 0).sum())))
    print("selection optimism (in-sample best - nested): {:+.4f}".format(insample.iloc[0] - nested.mean()))


if __name__ == "__main__":
    main()
