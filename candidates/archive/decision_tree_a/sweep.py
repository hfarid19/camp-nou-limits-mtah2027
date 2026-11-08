#!/usr/bin/env python3
"""Score (dead tree, team tree, decoder) settings on the DFL bench v2, leave-one-game-out.

    OMP_NUM_THREADS=2 python candidates/decision_tree_a/sweep.py '<json setting>' [...]

A setting is a JSON object:
    dead   "<sweep dir>" (e.g. dead_base_L200_d8), or "old" for the teammate's joint 3-class tree
           (decision_tree, $MTAH_WORK/arch/decision_tree/sweep/L200_d12, read-only) and its P(D)
    team   "<sweep dir>" (e.g. team_ext_L200_d10), or "old" for P(H) / (P(H) + P(A)) of that tree
    hw     half width (frames) of the moving mean of the team log-odds (0 = off)
    hwd    the same for the dead log-odds
    control_s   control.fold threshold (default 2.0)
    any decoder.CFG key (switch_cost, dead_cost, contrast_beta, ...)

It scores exactly what bench.py --dfl2 scores (bench.dfl2_prepared games x seeds 2027-2029,
possession_metrics.evaluate on bench.controlled_labels with T = 1, 2, 3 s), with the fold tree
that never saw the scored game, and computes the features on the fly (about 1 s per game).
"""

import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.insert(0, CAND)
sys.path.insert(0, HERE)
import bench                     # noqa: E402
import control                   # noqa: E402
import possession_metrics as pm  # noqa: E402
import pipeline as P             # noqa: E402

WORK = bench.WORK
SWEEP = os.path.join(WORK, "arch", "decision_tree_a", "sweep")
OLD = os.path.join(WORK, "arch", "decision_tree", "sweep", "L200_d12", "fold_tree_{}.npz")
MATCHES = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
JOBS = [(gi, m, s) for gi, m in enumerate(MATCHES) for s in bench.DFL2_SEEDS]


def _tree(path):
    with np.load(path) as z:
        t = {k: z[k] for k in z.files}
    t["names"] = [str(s) for s in t["names"]]
    return t


def run(args):
    (gi, match, seed), settings = args
    slots, slot_team, ball, gt, blocks, sign = bench.dfl2_prepared(match, seed)
    X, names, kinds, nbase = P.features(slots, slot_team, ball, sign, blocks, bench.tsd.PITCH, 25)
    perm, signs = P.fe.swap_index(names, kinds)
    Xs = X[:, perm] * signs
    lg = P.dec.long_gap_gated(slots, ball, 25, P.dec.CFG["gap_s"], P.dec.CFG["gap_speed_ms"])
    aux = dict(F=X[:, :nbase], names=names[:nbase], p_change=None, long_gap=lg, kick_restarts={})
    runs = [(int(a), int(b) + 1) for _, a, b in blocks]
    cache = {}

    def prob(kind, spec):
        if (kind, spec) not in cache:
            if spec == "old":
                t = _tree(OLD.format(gi))
                nf = len(t["names"])
                p = t["prob"][P.tw.leaf_index(t, X[:, :nf])]
                q = t["prob"][P.tw.leaf_index(t, Xs[:, :nf])][:, [1, 0, 2]]
                p = 0.5 * (p + q)
                cache[("dead", "old")] = p[:, 2]
                cache[("team", "old")] = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-12)
            else:
                t = _tree(os.path.join(SWEEP, spec, "fold_{}.npz".format(gi)))
                cache[(kind, spec)] = (P.sym_binary(t, X, Xs, 1, 1) if kind == "dead"
                                       else P.sym_binary(t, X, Xs, 0, 1))
        return cache[(kind, spec)]

    out = []
    for st in settings:
        pd = P.smooth_logit(prob("dead", st["dead"]), st.get("hwd", 0))
        ph = P.smooth_logit(prob("team", st["team"]), st.get("hw", 0))
        cfg = {k: v for k, v in st.items() if k in P.dec.CFG}
        codes = np.asarray(P.dec.decode(pd, ph, aux, blocks, len(gt), 25, cfg=cfg), np.int8)
        if st.get("control_s", 2.0):
            codes = control.fold(codes, st.get("control_s", 2.0), runs, 25)
        ctrls = [pm.evaluate(bench.controlled_labels(gt, T), codes, bootstrap_n=0)["spell_f1_mean"]
                 for T in (1, 2, 3)]
        r = pm.evaluate(gt, codes, bootstrap_n=0)
        out.append(dict(ctrl_mean=float(np.mean(ctrls)), ctrl=ctrls, raw=r["spell_f1_mean"],
                        poss=int(r["n_pred_spells"])))
    return out


def main():
    settings = [json.loads(a) for a in sys.argv[1:]]
    with mp.get_context("spawn").Pool(2) as pool:
        res = pool.map(run, [(j, settings) for j in JOBS])
    for i, st in enumerate(settings):
        rr = [r[i] for r in res]
        print("{:80s} ctrl_mean {:.4f} raw {:.4f} poss/seed {:.0f}".format(
            json.dumps(st), np.mean([r["ctrl_mean"] for r in rr]), np.mean([r["raw"] for r in rr]),
            sum(r["poss"] for r in rr) / 3), flush=True)


if __name__ == "__main__":
    main()
