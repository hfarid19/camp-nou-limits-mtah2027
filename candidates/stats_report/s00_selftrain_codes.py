"""Rebuild selftrain_ctrl's bench-v2 codes for all 21 game-seeds from saved offline outputs.

The live bench run of selftrain_ctrl saved codes for one game only (WMX). Its offline harness
(mtah-work/arch/selftrain_ctrl/offline) saved the self-trained model's P_st for all 21 game-seeds
(pst/lgb100_m12 = LightGBM 100 rounds, min_dist 12, the shipped setting). This script combines
them exactly as exp_combo.soft_add + fold (w = 1.5, fold 2.0 s), then checks agreement with the
3 live codes files. Output: $STATS/selftrain_codes/dfl2_<match>_<seed>_codes.npy
"""
import os
import sys

import numpy as np

WORK = "/path/to/mtah-work"
OFF = os.path.join(WORK, "arch/selftrain_ctrl/offline")
OUT = os.path.join(WORK, "stats/selftrain_codes")
sys.path.insert(0, OFF)
import common  # noqa: E402
import exp_combo  # noqa: E402

os.makedirs(OUT, exist_ok=True)
for gi, m in enumerate(common.MATCHES):
    for seed in common.SEEDS:
        g = common.load_game(gi, seed)
        P = exp_combo.load_P("lgb100_m12", g)
        codes = common.fold(exp_combo.soft_add(g, P, 1.5), g)
        np.save(os.path.join(OUT, "dfl2_{}_{}_codes.npy".format(m, seed)), codes.astype(np.int8))
        live = os.path.join(WORK, "probs/selftrain_ctrl/dfl2_{}_{}_codes.npy".format(m, seed))
        if os.path.isfile(live):
            lv = np.load(live)
            print(m, seed, "agreement with live codes: {:.5f}".format(float((lv == codes).mean())),
                  "ctrl live {:.4f} rebuilt {:.4f}".format(common.score(lv, g)["ctrl"],
                                                           common.score(codes, g)["ctrl"]))
        else:
            print(m, seed, "rebuilt")
