#!/usr/bin/env python3
"""MET-FT gate part 2: member alone (own codes), fine-tuned vs parent, paired.
  DFL fine4  (bench --dfl2 LOGO codes, probs/nn_pff_dfl/<name>; 21 game-seeds, SE over the 7 game means, wins)
  SkillCorner fine3 (probs/skc/<name>; 20 games, SE over games, wins)
plus possessions/90 (predicted and truth, raw labels). Gate: neither worse than the parent by more than 1 SE.

    python -I candidates/met_ft/stage2_report.py PARENT NAME     e.g. m5c_C192ls met_m5c
"""
import os
import sys

import multiprocessing.pool    # noqa: F401
import multiprocessing.queues  # noqa: F401
import numpy as np

CAND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(CAND, "nn_pff"))
import vote_offline as vo  # noqa: E402

PROBS = os.path.join(vo.WORK, "probs")


def p90(rows, info, key_blocks):
    n_pred = sum(r["n_pred"] for r in rows.values())
    n_gt = sum(r["n_gt"] for r in rows.values())
    mins = sum(sum(int(l) - int(f) + 1 for _, f, l in key_blocks[u]) for u in rows) / 25.0 / 60.0
    return n_pred / mins * 90, n_gt / mins * 90


def main():
    ref, name = sys.argv[1], sys.argv[2]
    ok = True
    info = vo.dfl_info()
    S = {}
    for m in (ref, name):
        S[m] = {}
        for u, (b, tg) in info.items():
            f = os.path.join(PROBS, "nn_pff_dfl", m, "dfl2_{}_{}_codes.npy".format(*u))
            if os.path.isfile(f):
                pred = np.load(f).astype(np.int8)
                r = vo.score_dfl(pred, tg)
                r["n_gt"] = vo.pm.evaluate(tg["raw"], pred, bootstrap_n=0)["n_gt_spells"]
                S[m][u] = r
    common = sorted(set(S[ref]) & set(S[name]))
    if common:
        d, se, w, n = vo.paired_dfl({u: S[name][u] for u in common}, {u: S[ref][u] for u in common}, "fine4")
        blocks = {u: info[u][0] for u in info}
        pr, gt = p90({u: S[ref][u] for u in common}, info, blocks)
        pn, _ = p90({u: S[name][u] for u in common}, info, blocks)
        print("DFL fine4 ({} game-seeds): {} {:.4f}  {} {:.4f}  diff {:+.4f} ± {:.4f} (wins {}/{})  poss/90 {:.0f} -> {:.0f} "
              "(raw truth {:.0f})".format(n, ref, np.mean([S[ref][u]["fine4"] for u in common]), name,
                                          np.mean([S[name][u]["fine4"] for u in common]), d, se, w, n, pr, pn, gt))
        for k in ("raw", "T0.5", "T1", "evt1"):
            dk, sk, wk, _ = vo.paired_dfl({u: S[name][u] for u in common}, {u: S[ref][u] for u in common}, k)
            print("   {:5s} {:+.4f} ± {:.4f} (wins {})".format(k, dk, sk, wk))
        dfl_ok = n == 21 and d >= -se
        print("   DFL gate (not worse by > 1 SE, 21 units): {}".format("PASS" if dfl_ok else "FAIL" if n == 21 else "INCOMPLETE"))
        ok &= dfl_ok
    else:
        print("DFL: no common units yet")
        ok = False
    sk = vo.skc_info()
    T = {}
    for m in (ref, name):
        T[m] = {}
        for g, (b, tg) in sk.items():
            f = os.path.join(PROBS, "skc", m, "skc2_{}_2027_codes.npy".format(g))
            if os.path.isfile(f):
                pred = np.load(f).astype(np.int8)
                r = {k: vo.pm.evaluate(tg[k], pred, bootstrap_n=0)["spell_f1_mean"] for k in ("raw", "T0.5", "T1")}
                r["fine3"] = float(np.mean(list(r.values())))
                ev = vo.pm.evaluate(tg["raw"], pred, bootstrap_n=0)
                r["n_pred"], r["n_gt"] = ev["n_pred_spells"], ev["n_gt_spells"]
                T[m][g] = r
    cg = sorted(set(T[ref]) & set(T[name]))
    if cg:
        dd = np.array([T[name][g]["fine3"] - T[ref][g]["fine3"] for g in cg])
        se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else float("nan")
        blocks = {g: sk[g][0] for g in sk}
        pr, gt = p90({g: T[ref][g] for g in cg}, sk, blocks)
        pn, _ = p90({g: T[name][g] for g in cg}, sk, blocks)
        print("SkillCorner fine3 ({} games): {} {:.4f}  {} {:.4f}  diff {:+.4f} ± {:.4f} (wins {}/{}, losses {})  "
              "poss/90 {:.0f} -> {:.0f} (raw truth {:.0f})".format(
                  len(cg), ref, np.mean([T[ref][g]["fine3"] for g in cg]), name, np.mean([T[name][g]["fine3"] for g in cg]),
                  dd.mean(), se, int((dd > 0).sum()), len(dd), int((dd < 0).sum()), pr, pn, gt))
        skc_ok = len(cg) == 20 and dd.mean() >= -se
        print("   SKC gate (not worse by > 1 SE, 20 games): {}".format("PASS" if skc_ok else "FAIL" if len(cg) == 20 else "INCOMPLETE"))
        ok &= skc_ok
    else:
        print("SkillCorner: no common games yet")
        ok = False
    print("GATE part 2 (DFL and SkillCorner): {}".format("PASS" if ok else "FAIL/INCOMPLETE"))


if __name__ == "__main__":
    main()
