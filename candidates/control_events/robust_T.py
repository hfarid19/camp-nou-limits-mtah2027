#!/usr/bin/env python3
"""Fold threshold for ensemble_vote's output against every label definition (README "Robustness").

    python -I candidates/control_events/robust_T.py

Rows: control.fold(prediction, T). Columns: control.fold(DFL, 1/2/3 s) labels and the event
labels with HOLD = 1, 2, 3 (main), 4 s, no hold, and OtherBallAction counted as control.
Cells: mean spellF1 over 7 games x 3 seeds; last columns: worst-case regret against each
column's own best T, and the mean.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_events as S          # noqa: E402
import control                    # noqa: E402
import possession_metrics as pm   # noqa: E402

V = os.path.join(S.WORK, "arch", "control_events", "variants")
COLS = [("ctrl1", 1), ("ctrl2", 2), ("ctrl3", 3), ("evt_hold1", os.path.join(V, "hold1")),
        ("evt_hold2", os.path.join(V, "hold2")), ("evt_hold3", S.LABELS), ("evt_hold4", os.path.join(V, "hold4")),
        ("evt_nohold", os.path.join(V, "holdinf")), ("evt_touch", os.path.join(V, "touch"))]
TS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


def main(slug="ensemble_vote"):
    matches = sorted(f[:-4] for f in os.listdir(S.NPZ) if f.endswith(".npz"))
    M = np.zeros((len(TS), len(COLS)))
    for m in matches:
        gt = S.raw_gt(m)
        runs = pm.valid_runs(gt)
        labs = [control.fold(gt, c, runs) if isinstance(c, int) else S.event_gt(m, c) for _, c in COLS]
        for s in (2027, 2028, 2029):
            pred = np.load(os.path.join(S.WORK, "probs", slug, "dfl2_{}_{}_codes.npy".format(m, s)))
            for i, T in enumerate(TS):
                f = control.fold(pred, T, runs)
                for j, lab in enumerate(labs):
                    M[i, j] += pm.evaluate(lab, f, bootstrap_n=0)["spell_f1_mean"] / (3 * len(matches))
    reg = M.max(0) - M
    print("{:6s} ".format("T") + " ".join("{:>10s}".format(c) for c, _ in COLS) + "  worst_reg  mean_all")
    for i, T in enumerate(TS):
        print("{:6.2f} ".format(T) + " ".join("{:10.4f}".format(v) for v in M[i]) +
              "  {:9.4f}  {:8.4f}".format(reg[i].max(), M[i].mean()))
    print("best T per column:", {c: TS[int(np.argmax(M[:, j]))] for j, (c, _) in enumerate(COLS)})


if __name__ == "__main__":
    main(*sys.argv[1:])
