#!/usr/bin/env python3
"""Offline v4-style votes of saved member codes on DFL (dfl2, LOGO codes) and PFF (out-of-fold codes).

    python candidates/nn_pff/vote_offline.py --set NAME=m1,m2,m3 [--set ...] --ref-dfl ensemble_ctrl_v4 \
        --ref-pff ensemble_ctrl_v4 [--grid default] [--select pff|dfl|both]

A member m is "dfl_dir|pff_dir" (dfl_dir under $MTAH_WORK/probs, pff_dir under probs/nn_pff or
probs/pff) or a single name used for both. Every DFL code file is a leave-one-game-out prediction
(bench --dfl2) and every PFF code file of an nn_pff member is out-of-fold (run_pff.py); DFL-only
members never saw PFF. A vote = ensemble_core.hard_vote (tie-break = member order) ->
merge_short_v2 (smooth s, off if 0) -> control.fold (T), exactly ensemble_ctrl_v4's _run (prekick_dead
is a no-op without tags). Single-member sets are scored as the member's codes -> smooth -> fold.

Scores: DFL fine4 (score_sel2: raw flag, T0.5, T1, event HOLD1) per game-seed; PFF per game
(bench_pff.score_codes): raw = touch labels, fine3 = mean(raw, T0.5, T1).
Nested selection: the (smooth, fold) of each set is chosen on the other DFL games (7 folds) and
on the other PFF selection folds (4 folds); --select picks the criterion (pff: PFF fine3, dfl: DFL
fine4, both: their mean, standardised by nothing). Reported: nested scores, paired differences vs
the reference codes (SE over games, wins).
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [os.path.join(CAND, "pff"), os.path.join(CAND, "ensemble_ctrl_v4"), CAND, ROOT,
                os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import control                   # noqa: E402
import ensemble_core as ec       # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
ARCH = os.path.join(WORK, "arch", "nn_pff")
CACHE = os.path.join(ARCH, "vote_cache")
SEEDS = (2027, 2028, 2029)
GRID = [(s, T) for s in (0.0, 1.0) for T in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)]
_LDE = []


def lde():
    if not _LDE:
        import members
        _LDE.append(members.load("learned_dfl_evidence"))
    return _LDE[0]


def dfl_units():
    ms = sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "idsse", "npz")) if f.endswith(".npz"))
    return [(m, s) for m in ms for s in SEEDS]


def pff_games():
    f = json.load(open(os.path.join(ARCH, "selection_folds.json")))["folds"]
    return {g: k for k, fold in enumerate(f) for g in fold}


def _cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.isfile(p):
        return pickle.load(open(p, "rb"))
    v = fn()
    pickle.dump(v, open(p, "wb"))
    return v


def dfl_info():
    def f():
        import bench
        import score_sel2
        out = {}
        for m, s in dfl_units():
            blocks = bench.dfl2_prepared(m, s)[4]
            out[(m, s)] = (blocks, score_sel2.targets(m, s, "dfl2"))
        return out
    return _cached("dfl2_targets", f)


def pff_info():
    def f():
        import bench_pff
        out = {}
        for g in pff_games():
            blocks = bench_pff.pff2_prepared(g)[4]
            out[g] = (blocks, bench_pff.targets(g))
        return out
    return _cached("pff_targets", f)


def met_info():
    """Metrica dev (Sample Games 1, 2 x seeds 2027, 2028; practice labels): report-only domain."""
    import bench
    out = {}
    for g in bench.DEV:
        for sd in bench.SEEDS:
            z = bench.prepared(g, sd)
            out[(g, sd)] = (z[4], z[3])
    return out


def met_codes(dname, unit):
    for root in (os.path.join(WORK, "probs", "nn_pff_metrica"), os.path.join(WORK, "probs", "metrica_dev"),
                 os.path.join(WORK, "probs")):
        f = os.path.join(root, dname, "dev_{}_{}_codes.npy".format(*unit))
        if os.path.isfile(f):
            return np.load(f).astype(np.int8)
    raise SystemExit("no Metrica dev codes for {} {}".format(dname, unit))


def met_scores(members, cfg):
    """{unit: spellF1} of the vote of Metrica dev member codes (member names = folder names)."""
    out = {}
    for u, (blocks, gt) in met_info().items():
        cl = [met_codes(m, u) for m in members]
        out[u] = pm.evaluate(gt, vote(cl, blocks, *cfg), bootstrap_n=0)["spell_f1_mean"]
    return out


def skc_info(games=None):
    """SkillCorner A-League bench (20 games, degrade_v2 seed 2027). Written as a report-only check, but it
    became a selection bench: it decided v6's rule and was a gate (and half the pick criterion) in rounds
    4-5. Members trained with SKC games must be scored on the 5 SKC holdout games only."""
    def f():
        sys.path.insert(0, os.path.join(CAND, "skc"))
        import bench_skc
        return {g: (bench_skc.skc2_prepared(g)[4], {k: v for k, v in bench_skc.targets(g).items()
                                                    if k in ("raw", "T0.5", "T1")}) for g in bench_skc.all_games()}
    info = _cached("skc_targets", f)
    return info if games is None else {g: info[g] for g in games}


def skc_scores(members, cfg, games=None):
    """{game: dict(raw, fine3)} of the vote of SKC member codes (probs/skc/<member>)."""
    out = {}
    for g, (blocks, tg) in skc_info(games).items():
        cl = [np.load(os.path.join(WORK, "probs", "skc", m, "skc2_{}_2027_codes.npy".format(g))).astype(np.int8)
              for m in members]
        pred = vote(cl, blocks, *cfg)
        r = {k: pm.evaluate(tg[k], pred, bootstrap_n=0)["spell_f1_mean"] for k in ("raw", "T0.5", "T1")}
        r["fine3"] = float(np.mean([r[k] for k in ("raw", "T0.5", "T1")]))
        out[g] = r
    return out


def find_pff(name):
    for root in (os.path.join(WORK, "probs", "nn_pff"), os.path.join(WORK, "probs", "pff")):
        if os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    raise SystemExit("no PFF codes for " + name)


def member_codes(member, domain, unit):
    dname, pname = member.split("|") if "|" in member else (member, member)
    if domain == "dfl":
        m, s = unit
        return np.load(os.path.join(WORK, "probs", dname, "dfl2_{}_{}_codes.npy".format(m, s))).astype(np.int8)
    return np.load(os.path.join(find_pff(pname), "pff2_{}_2027_codes.npy".format(unit))).astype(np.int8)


def vote(codes_list, blocks, smooth, T):
    sm = dict(min_poss_s=smooth, min_dead_s=smooth) if smooth > 0 else None
    if len(codes_list) == 1 and sm is None:
        v = codes_list[0].copy()
    else:
        v = ec.hard_vote(codes_list, blocks, lde(), 25, None, sm)
    runs = [(int(f), int(l) + 1) for _, f, l in blocks]
    return control.fold(np.asarray(v, np.int8), T, runs, 25)


def score_dfl(pred, tg):
    r = {k: pm.evaluate(g, pred, bootstrap_n=0)["spell_f1_mean"] for k, g in tg.items()
         if k in ("raw", "T0.5", "T1", "evt1")}
    r["fine4"] = float(np.mean([r[k] for k in ("raw", "T0.5", "T1", "evt1")]))
    r["n_pred"] = pm.evaluate(tg["raw"], pred, bootstrap_n=0)["n_pred_spells"]
    return r


def score_pff(pred, tg):
    r = {k: pm.evaluate(tg[k], pred, bootstrap_n=0)["spell_f1_mean"] for k in ("raw", "T0.5", "T1", "seq", "nore")}
    r["fine3"] = float(np.mean([r[k] for k in ("raw", "T0.5", "T1")]))
    ev = pm.evaluate(tg["raw"], pred, bootstrap_n=0)
    r["macroF1"], r["n_pred"] = ev["macro_f1"], ev["n_pred_spells"]
    return r


def evaluate_set(members, grid, domains=("dfl", "pff")):
    """-> {domain: {cfg: {unit: scores}}}"""
    out = {}
    if "dfl" in domains:
        info = dfl_info()
        out["dfl"] = {}
        for u, (blocks, tg) in info.items():
            cl = [member_codes(m, "dfl", u) for m in members]
            for cfg in grid:
                out["dfl"].setdefault(cfg, {})[u] = score_dfl(vote(cl, blocks, *cfg), tg)
    if "pff" in domains:
        info = pff_info()
        out["pff"] = {}
        for g, (blocks, tg) in info.items():
            cl = [member_codes(m, "pff", g) for m in members]
            for cfg in grid:
                out["pff"].setdefault(cfg, {})[g] = score_pff(vote(cl, blocks, *cfg), tg)
    return out


def ref_scores(dfl_dir, pff_dir):
    out = {"dfl": {}, "pff": {}}
    if dfl_dir:
        for u, (blocks, tg) in dfl_info().items():
            out["dfl"][u] = score_dfl(np.load(os.path.join(WORK, "probs", dfl_dir, "dfl2_{}_{}_codes.npy".format(*u))).astype(np.int8), tg)
    if pff_dir:
        for g, (blocks, tg) in pff_info().items():
            out["pff"][g] = score_pff(np.load(os.path.join(find_pff(pff_dir), "pff2_{}_2027_codes.npy".format(g))).astype(np.int8), tg)
    return out


def crit(S, cfg, units_dfl, units_pff, select):
    a = np.mean([S["dfl"][cfg][u]["fine4"] for u in units_dfl]) if select in ("dfl", "both") else 0.0
    b = np.mean([S["pff"][cfg][g]["fine3"] for g in units_pff]) if select in ("pff", "both") else 0.0
    return a + b


def nested(S, grid, select):
    """Nested choice of cfg: for each DFL game (resp. PFF fold) the cfg maximising the criterion on
    the other DFL games and all PFF games (resp. other PFF folds and all DFL games) -- no unit is
    ever scored by a cfg chosen on itself. -> ({domain: {unit: scores}}, picks)."""
    fold = pff_games()
    du = sorted(S["dfl"][grid[0]])
    pu = sorted(S["pff"][grid[0]])
    res = {"dfl": {}, "pff": {}}
    picks = {"dfl": {}, "pff": {}}
    for m in sorted({u[0] for u in du}):
        tr = [u for u in du if u[0] != m]
        best = max(grid, key=lambda c: crit(S, c, tr, pu, select))
        picks["dfl"][m] = best
        for u in du:
            if u[0] == m:
                res["dfl"][u] = S["dfl"][best][u]
    for k in range(4):
        tr = [g for g in pu if fold[g] != k]
        best = max(grid, key=lambda c: crit(S, c, du, tr, select))
        picks["pff"][k] = best
        for g in pu:
            if fold[g] == k:
                res["pff"][g] = S["pff"][best][g]
    return res, picks


def paired_dfl(a, b, key):
    common = sorted(set(a) & set(b))
    games = sorted({m for m, _ in common})
    gm = np.array([np.mean([a[u][key] - b[u][key] for u in common if u[0] == g]) for g in games])
    wins = sum(a[u][key] > b[u][key] for u in common)
    return gm.mean(), gm.std(ddof=1) / np.sqrt(len(gm)), wins, len(common)


def paired_pff(a, b, key):
    common = sorted(set(a) & set(b))
    d = np.array([a[g][key] - b[g][key] for g in common])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)


def report(name, res, ref, picks=None):
    d, p = res["dfl"], res["pff"]
    line = "{:34s}".format(name[:34])
    if d:
        line += " DFL fine4 {:.4f} raw {:.4f} poss {:5d}".format(np.mean([r["fine4"] for r in d.values()]),
                                                            np.mean([r["raw"] for r in d.values()]),
                                                            int(sum(r["n_pred"] for r in d.values())))
        if ref["dfl"]:
            mu, se, w, n = paired_dfl(d, ref["dfl"], "fine4")
            line += " ({:+.4f}±{:.4f} {}/{})".format(mu, se, w, n)
    if p:
        line += " | PFF touch {:.4f} fine3 {:.4f} seq {:.4f} poss {:5d}".format(
            np.mean([r["raw"] for r in p.values()]), np.mean([r["fine3"] for r in p.values()]),
            np.mean([r["seq"] for r in p.values()]), int(sum(r["n_pred"] for r in p.values())))
        if ref["pff"]:
            for k in ("raw", "fine3"):
                mu, se, w, n = paired_pff(p, ref["pff"], k)
                line += " {} {:+.4f}±{:.4f} ({}/{})".format(k, mu, se, w, n)
    print(line)
    if picks:
        print("    picks DFL {} | PFF {}".format(sorted(set(picks["dfl"].values())), sorted(set(picks["pff"].values()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="NAME=m1,m2,...")
    ap.add_argument("--ref-dfl", default="ensemble_ctrl_v4")
    ap.add_argument("--ref-pff", default="ensemble_ctrl_v4")
    ap.add_argument("--select", default="pff", choices=["pff", "dfl", "both"])
    ap.add_argument("--domains", default="dfl,pff")
    ap.add_argument("--grid-all", action="store_true", help="print every grid point")
    a = ap.parse_args()
    doms = tuple(a.domains.split(","))
    ref = ref_scores(a.ref_dfl if "dfl" in doms else None, a.ref_pff if "pff" in doms else None)
    report("REF dfl={} pff={}".format(a.ref_dfl, a.ref_pff), ref, {"dfl": {}, "pff": {}})
    for s in a.set:
        name, ms = s.split("=", 1)
        S = evaluate_set(ms.split(","), GRID, doms)
        if a.grid_all or len(doms) < 2:
            for cfg in GRID:
                report("  {} s{:g} T{:g}".format(name, *cfg), {d: S.get(d, {}).get(cfg, {}) for d in ("dfl", "pff")}, ref)
        if len(doms) == 2:
            report("  {} v4 rule (s1 T1.25)".format(name), {d: S[d][(1.0, 1.25)] for d in doms}, ref)
            res, picks = nested(S, GRID, a.select)
            report("  {} NESTED[{}]".format(name, a.select), res, ref, picks)


if __name__ == "__main__":
    main()
