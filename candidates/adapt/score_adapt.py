#!/usr/bin/env python3
"""Paired scores of member-code sets under the v6 rule (vote -> merge_short 1 s -> control.fold 1.25 s;
prekick_dead is a no-op on the benches) on DFL bench v2 (fine4, LOGO codes, SE over the 7 games),
SkillCorner A-League (fine3, 20 games) and Metrica dev, vs v6 and vs lsw.

    python candidates/adapt/score_adapt.py --set NAME=m1;m2;m3 ... [--nested NAME1,NAME2,...]

A member is a directory under $MTAH_WORK/probs/adapt (e.g. o_ls05__all_a50, any folder holding
dfl2_/skc2_/dev_ codes) or "orig:<cfg>" (the members' saved bench codes: probs/nn_pff_dfl/<cfg>,
probs/skc/<cfg>, probs/nn_pff_metrica/<cfg>). --nested: the per-DFL-game choice among those sets made
on the other six DFL games (DFL fine4); SKC / Metrica get the choice made on all 7 DFL games.
"""

import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "nn_pff")]
import vote_offline as vo  # noqa: E402

WORK = vo.WORK
RULE = (1.0, 1.25)
V6 = "orig:t5c;orig:m5c;orig:radar_pff"
LSW = "orig:o_ls05;orig:m5c_W;orig:radar_pff"


def path(member, dom, tag):
    if member.startswith("orig:"):
        c = member[5:]
        root = {"dfl": "nn_pff_dfl", "skc": "skc", "met": "nn_pff_metrica"}[dom]
        return os.path.join(WORK, "probs", root, c, tag + "_codes.npy")
    if member.startswith("/"):
        return os.path.join(member, tag + "_codes.npy")
    return os.path.join(WORK, "probs", "adapt", member, tag + "_codes.npy")


def load(member, dom, tag):
    return np.load(path(member[6:] if member.startswith("final:") else member, dom, tag)).astype(np.int8)


def vote(ms, dom, tag, blocks):
    """A set "final:<dir>" is already a complete decode (e.g. MBR): its codes are scored as they are."""
    if len(ms) == 1 and ms[0].startswith("final:"):
        return load(ms[0], dom, tag)
    return vo.vote([load(m, dom, tag) for m in ms], blocks, *RULE)


def n_poss(pred, blocks):
    """Possession spells (H / A runs) and graded frames inside the blocks."""
    runs = [(int(f), int(l) + 1) for _, f, l in blocks]
    return vo.pm.possession_stats(pred, runs)["n_spells"], sum(b - a for a, b in runs)


def per90(rows):
    n = sum(r["poss"][0] for r in rows.values())
    fr = sum(r["poss"][1] for r in rows.values())
    return 90 * 60 * 25 * n / fr


def skc_fine3(pred, tg):
    r = {k: vo.pm.evaluate(tg[k], pred, bootstrap_n=0)["spell_f1_mean"] for k in ("raw", "T0.5", "T1")}
    r["fine3"] = float(np.mean(list(r.values())))
    return r


def scores(ms):
    D, K, M = {}, {}, {}
    for u, (blocks, tg) in vo.dfl_info().items():
        pred = vote(ms, "dfl", "dfl2_{}_{}".format(*u), blocks)
        D[u] = vo.score_dfl(pred, tg)
        D[u]["poss"] = n_poss(pred, blocks)
    for g, (blocks, tg) in vo.skc_info().items():
        pred = vote(ms, "skc", "skc2_{}_2027".format(g), blocks)
        K[g] = skc_fine3(pred, tg)
        K[g]["poss"] = n_poss(pred, blocks)
    for u, (blocks, gt) in vo.met_info().items():
        pred = vote(ms, "met", "dev_{}_{}".format(*u), blocks)
        M[u] = {"f1": vo.pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"], "poss": n_poss(pred, blocks)}
    return D, K, M


EXTRA = []
T_GRID = [round(0.25 + 0.05 * i, 2) for i in range(76)]          # 0.25 .. 4.0 s
_V6T = {}


def v6_matched(K):
    """Prereg addendum: v6 re-folded per SKC game to the candidate's possession count (fold T on a 0.05 s
    grid, the T whose count is closest; ties -> nearest 1.25 s). -> {game: fine3 of matched v6}."""
    out = {}
    for g, (blocks, tg) in vo.skc_info().items():
        if g not in _V6T:
            cl = [load(m, "skc", "skc2_{}_2027".format(g)) for m in V6.split(";")]
            _V6T[g] = {T: vo.vote(cl, blocks, RULE[0], T) for T in T_GRID}
        target = K[g]["poss"][0]
        T = min(T_GRID, key=lambda t: (abs(n_poss(_V6T[g][t], blocks)[0] - target), abs(t - RULE[1])))
        out[g] = skc_fine3(_V6T[g][T], tg)
        out[g]["T"] = T
    return out


def line(name, S, R6, RL, matched=True):
    D, K, M = S
    out = "{:28s} DFL {:.4f} ".format(name[:28], np.mean([r["fine4"] for r in D.values()]))
    for R in (R6, RL):
        mu, se, w, n = vo.paired_dfl(D, R[0], "fine4")
        out += "{:+.4f}±{:.4f} ({:2d}/{}) ".format(mu, se, w, n)
    out += "| SKC {:.4f} ".format(np.mean([r["fine3"] for r in K.values()]))
    for R in (R6, RL):
        mu, se, w, n = vo.paired_pff(K, R[1], "fine3")
        out += "{:+.4f}±{:.4f} ({:2d}/{}) ".format(mu, se, w, n)
    m = np.mean([r["f1"] for r in M.values()])
    out += "| Met {:.4f} {:+.4f} {:+.4f}".format(m, m - np.mean([r["f1"] for r in R6[2].values()]),
                                                m - np.mean([r["f1"] for r in RL[2].values()]))
    out += " | poss/90 DFL {:.0f} SKC {:.0f} Met {:.0f}".format(per90(D), per90(K), per90(M))
    for n2, R2 in EXTRA:
        mu, se, w, n = vo.paired_dfl(D, R2[0], "fine4")
        out += " | vs {} DFL {:+.4f}±{:.4f}".format(n2, mu, se)
        mu, se, w, n = vo.paired_pff(K, R2[1], "fine3")
        out += " SKC {:+.4f}±{:.4f} ({}/{})".format(mu, se, w, n)
        out += " Met {:+.4f}".format(np.mean([r["f1"] for r in M.values()]) - np.mean([r["f1"] for r in R2[2].values()]))
    if matched:
        V = v6_matched(K)
        mu, se, w, n = vo.paired_pff(K, V, "fine3")
        out += " | SKC vs v6@matched-poss {:+.4f}±{:.4f} ({:2d}/{})".format(mu, se, w, n)
    return out


def nested(named):
    """named: {name: S}. Per DFL game: the set with the best mean fine4 on the other games."""
    games = sorted({u[0] for u in next(iter(named.values()))[0]})
    D, picks = {}, {}
    for g in games:
        best = max(named, key=lambda k: np.mean([r["fine4"] for u, r in named[k][0].items() if u[0] != g]))
        picks[g] = best
        D.update({u: r for u, r in named[best][0].items() if u[0] == g})
    allbest = max(named, key=lambda k: np.mean([r["fine4"] for r in named[k][0].values()]))
    return (D, named[allbest][1], named[allbest][2]), picks, allbest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--nested", action="append", default=[], help="LABEL=name1,name2,...")
    ap.add_argument("--ref2", default=None, help="NAME=set: a third reference (paired column appended)")
    a = ap.parse_args()
    R6, RL = scores(V6.split(";")), scores(LSW.split(";"))
    if a.ref2:
        n2, s2 = a.ref2.split("=", 1)
        EXTRA.append((n2, scores(s2.split(";"))))
    print("{:28s} {:>10s} {:>27s} {:>27s} | ... (paired vs v6, then vs lsw; SE over DFL games / SKC games)".format(
        "set", "DFL fine4", "vs v6", "vs lsw"))
    print(line("v6", R6, R6, RL))
    print(line("lsw", RL, R6, RL))
    named = {}
    for s in a.set:
        n, ms = s.split("=", 1)
        named[n] = scores(ms.split(";"))
        print(line(n, named[n], R6, RL), flush=True)
    for spec in a.nested:
        lab, names = spec.split("=", 1)
        S, picks, allbest = nested({k: named[k] for k in names.split(",")})
        print(line("NESTED " + lab, S, R6, RL))
        print("    picks per DFL game: {} | SKC/Metrica use {}".format(
            ", ".join("{}:{}".format(g[-3:], p) for g, p in sorted(picks.items())), allbest))


if __name__ == "__main__":
    main()
