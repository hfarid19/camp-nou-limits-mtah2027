#!/usr/bin/env python3
"""Audit of candidates/decision_tree's evaluation and interpretability claims (read-only on its files).

    OMP_NUM_THREADS=1 python candidates/decision_tree_b/audit.py [--part all|control|ensemble|perm|rules]

Reads the teammate's fold trees ($MTAH_WORK/arch/decision_tree/folds), its saved bench codes
($MTAH_WORK/probs/decision_tree{,/raw}) and the other members' saved bench-v2 codes; recomputes
features with learned_dfl_evidence/features.py on the bench's own dfl2_prepared games. Every
number is on DFL game gi scored by a model that never saw game gi. Writes
$MTAH_WORK/arch/decision_tree_b/audit/audit.json and prints a report.

- control:  nested choice of the control-fold threshold T (raw decoded codes, folded afterwards).
- ensemble: frame majority vote temporal_tcn + handover_mstcn + X (ties to temporal_tcn), then
            control.fold 2 s; paired differences over games.
- perm:     held-out permutation importance of the tree's 194 inputs (frame accuracy and log loss on
            the scored game, fold tree), next to the fold trees' gini importance; then the
            decode-level drop in ctrl_mean for the top features.
- rules:    purity and coverage of the README's quoted rules on the held-out bench frames.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
DT = os.path.join(WORK, "arch", "decision_tree")
OUT = os.path.join(WORK, "arch", "decision_tree_b", "audit")
PROBS = os.path.join(WORK, "probs")
SEEDS = (2027, 2028, 2029)


def setup():
    if CAND not in sys.path:
        sys.path.insert(0, CAND)


def teammate_pipeline():
    """candidates/decision_tree/pipeline.py under a unique module name (read-only use)."""
    import importlib.util
    name = "audit_decision_tree_pipeline"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(CAND, "decision_tree", "pipeline.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def matches():
    return sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "idsse", "npz")) if f.endswith(".npz"))


def paired(a, b):
    """a, b: (games,) per-game means. Mean difference, SE over games, wins."""
    d = np.asarray(a) - np.asarray(b)
    return dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), wins=int((d > 0).sum()),
                n=len(d))


def ctrl_mean(gt, codes):
    import bench
    return float(np.mean([bench.pm.evaluate(bench.controlled_labels(gt, T), codes, bootstrap_n=0)
                          ["spell_f1_mean"] for T in (1, 2, 3)]))


def codes(slug, match, seed, sub=""):
    return np.load(os.path.join(PROBS, slug, sub, "dfl2_{}_{}_codes.npy".format(match, seed)))


# ------------------------------------------------------------------ control threshold, nested
def part_control():
    setup()
    import bench
    import control
    Ts = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
    ms = matches()
    S = np.zeros((len(Ts), len(ms)))
    for gi, m in enumerate(ms):
        for seed in SEEDS:
            gt, blocks = bench.dfl2_prepared(m, seed)[3:5]
            runs = [(a, b + 1) for _, a, b in blocks]
            raw = codes("decision_tree", m, seed, "raw")
            for k, T in enumerate(Ts):
                S[k, gi] += ctrl_mean(gt, control.fold(raw, T, runs)) / len(SEEDS)
    nested = []
    for gi in range(len(ms)):
        oth = [j for j in range(len(ms)) if j != gi]
        k = int(np.argmax(S[:, oth].mean(1)))
        nested.append((Ts[k], S[k, gi]))
    res = dict(T=list(Ts), mean=S.mean(1).tolist(), nested=float(np.mean([v for _, v in nested])),
               nested_picks=[t for t, _ in nested],
               d2_vs_best_other=paired(S[Ts.index(2.0)], S[int(np.argsort(-S.mean(1))[1])]))
    print("control fold T for the tree (raw decode, folded): " +
          ", ".join("{:g}: {:.4f}".format(t, v) for t, v in zip(Ts, res["mean"])))
    print("  nested T choice: picks {} -> {:.4f}".format(res["nested_picks"], res["nested"]))
    return res


# ------------------------------------------------------------------ ensemble claim
def part_ensemble():
    setup()
    import bench
    import control
    pm = bench.pm
    ms = matches()
    variants = {
        "tcn+mstcn+tree(raw)": ("temporal_tcn", "handover_mstcn", ("decision_tree", "raw")),
        "tcn+mstcn+tree(fold)": ("temporal_tcn", "handover_mstcn", ("decision_tree", "")),
        "tcn+mstcn+lde": ("temporal_tcn", "handover_mstcn", ("learned_dfl_evidence", "")),
        "tcn+mstcn+rules_b(raw)": ("temporal_tcn", "handover_mstcn", ("decision_tree_b", "raw")),
        "tcn alone": ("temporal_tcn",),
        "tcn+mstcn (tie->tcn)": ("temporal_tcn", "handover_mstcn"),
    }
    per = {k: np.zeros(len(ms)) for k in variants}
    per["ensemble_ctrl (saved)"] = np.zeros(len(ms))
    agree = {}
    for gi, m in enumerate(ms):
        for seed in SEEDS:
            gt, blocks = bench.dfl2_prepared(m, seed)[3:5]
            runs = [(a, b + 1) for _, a, b in blocks]
            for name, members in variants.items():
                cl = []
                for mem in members:
                    slug, sub = mem if isinstance(mem, tuple) else (mem, "")
                    path = os.path.join(PROBS, slug, sub, "dfl2_{}_{}_codes.npy".format(m, seed))
                    if not os.path.isfile(path):
                        cl = None
                        break
                    cl.append(np.load(path))
                if cl is None:
                    per[name][gi] = np.nan
                    continue
                V = np.zeros((len(gt), 3))
                for k, c in enumerate(cl):
                    for lab in (pm.H, pm.A, pm.D):
                        V[:, lab] += (c == lab) * (1.0 + 1e-3 * (len(cl) - k))
                lab = np.argmax(V, 1).astype(np.int8)
                out = np.full(len(gt), pm.X, np.int8)
                for a, b in runs:
                    out[a:b] = lab[a:b]
                per[name][gi] += ctrl_mean(gt, control.fold(out, 2.0, runs)) / len(SEEDS)
            per["ensemble_ctrl (saved)"][gi] += ctrl_mean(gt, codes("ensemble_ctrl", m, seed)) / len(SEEDS)
            g = gt != pm.X
            t, l = codes("decision_tree", m, seed, "raw"), codes("learned_dfl_evidence", m, seed)
            agree.setdefault("tree_vs_lde_disagree", []).append(float((t[g] != l[g]).mean()))
    res = {k: dict(mean=float(np.nanmean(v)), per_game=v.tolist()) for k, v in per.items()}
    for k, v in per.items():
        print("  {:26s} ctrl_mean {:.4f}".format(k, np.nanmean(v)))
    pairs = [("tcn+mstcn+tree(raw)", "tcn+mstcn+lde"), ("tcn+mstcn+tree(fold)", "tcn+mstcn+lde"),
             ("tcn+mstcn+tree(raw)", "tcn+mstcn (tie->tcn)"), ("tcn+mstcn+lde", "tcn+mstcn (tie->tcn)"),
             ("tcn+mstcn+tree(raw)", "ensemble_ctrl (saved)"), ("tcn+mstcn+rules_b(raw)", "tcn+mstcn+tree(raw)")]
    res["paired"] = {}
    for a, b in pairs:
        if np.isnan(per[a]).any() or np.isnan(per[b]).any():
            continue
        p = paired(per[a], per[b])
        res["paired"]["{} - {}".format(a, b)] = p
        print("  {} - {}: {:+.4f} (SE {:.4f}, {}/{} games)".format(a, b, p["diff"], p["se"], p["wins"], p["n"]))
    res["tree_vs_lde_disagree"] = float(np.mean(agree["tree_vs_lde_disagree"]))
    print("  tree (raw) vs LDE frame disagreement on graded frames: {:.3f}".format(res["tree_vs_lde_disagree"]))
    return res


# ------------------------------------------------------------------ permutation importance
def _game_data(match, seed, stride=None):
    setup()
    import bench
    dtp = teammate_pipeline()
    slots, slot_team, ball, gt, blocks, sign = bench.dfl2_prepared(match, seed)
    # fe.compute directly: pipeline.features memoises on id(array), which can return another
    # seed's features once a freed array's id is reused (see README, audit item 6)
    F, names, kinds = dtp.fe.compute(slots, slot_team, ball, dtp.fe.frame_sign(len(ball), sign, blocks),
                                     bench.tsd.PITCH, 25)
    return slots, ball, gt, blocks, F, names, kinds


def _tree(gi):
    z = np.load(os.path.join(DT, "folds", "fold_tree_{}.npz".format(gi)))
    t = {k: z[k] for k in z.files}
    t["names"] = [str(v) for v in t["names"]]
    return t


def _frame_scores(p, y):
    acc = float((np.argmax(p, 1) == y).mean())
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1))))
    return acc, ll


def perm_frame_job(args):
    gi, match = args
    setup()
    import bench
    dtp = teammate_pipeline()
    pm = bench.pm
    tree = _tree(gi)
    rng = np.random.default_rng(gi)
    base, drops_acc, drops_ll, val_only = [], [], [], []
    for seed in SEEDS:
        _, _, gt, _, F, names, kinds = _game_data(match, seed)
        idx = np.where(gt != pm.X)[0][::5]
        Fs, y = F[idx].copy(), gt[idx].astype(int)
        a0, l0 = _frame_scores(dtp.sym_proba(tree, Fs, names, kinds), y)
        base.append((a0, l0))
        da, dl, dv = [], [], []
        for j in range(Fs.shape[1]):
            col = Fs[:, j].copy()
            Fs[:, j] = col[rng.permutation(len(col))]
            a, l = _frame_scores(dtp.sym_proba(tree, Fs, names, kinds), y)
            da.append(a0 - a)
            dl.append(l - l0)
            ok = np.where(~np.isnan(col))[0]                # permute values, keep the NaN pattern
            Fs[:, j] = col
            Fs[ok, j] = col[ok][rng.permutation(len(ok))]
            a, _ = _frame_scores(dtp.sym_proba(tree, Fs, names, kinds), y)
            dv.append(a0 - a)
            Fs[:, j] = col
        drops_acc.append(da)
        drops_ll.append(dl)
        val_only.append(dv)
    return gi, np.mean(base, 0).tolist(), np.mean(drops_acc, 0).tolist(), np.mean(drops_ll, 0).tolist(), \
        np.mean(val_only, 0).tolist()


def perm_decode_job(args):
    gi, match, feats = args
    setup()
    import bench
    dtp = teammate_pipeline()
    pm = bench.pm
    tree = _tree(gi)
    rng = np.random.default_rng(100 + gi)
    out = {f: 0.0 for f in ["__base__"] + feats}
    for seed in SEEDS:
        slots, ball, gt, blocks, F, names, kinds = _game_data(match, seed)
        lg = dtp.dec.long_gap_gated(slots, ball, 25, dtp.dec.CFG["gap_s"], dtp.dec.CFG["gap_speed_ms"])
        graded = np.where(gt != pm.X)[0]
        for f in ["__base__"] + feats:
            Fp = F
            if f != "__base__":
                Fp = F.copy()
                j = names.index(f)
                Fp[graded, j] = F[graded, j][rng.permutation(len(graded))]
            p = dtp.sym_proba(tree, Fp, names, kinds)
            pred = dtp.decode(p, F, names, lg, {}, blocks, len(gt), 25, control_s=2.0)  # decoder aux unpermuted
            out[f] += ctrl_mean(gt, pred) / len(SEEDS)
    return gi, out


def gini_importance(t):
    w = t["count"].sum(1)
    imp = np.zeros(len(t["names"]))
    for k in np.where(t["feature"] >= 0)[0]:
        lft, rgt = t["left"][k], t["right"][k]
        imp[t["feature"][k]] += (w[k] * t["impurity"][k] - w[lft] * t["impurity"][lft]
                                 - w[rgt] * t["impurity"][rgt])
    return imp / imp.sum()


def part_perm(jobs):
    ms = matches()
    names = _tree(0)["names"]
    gini = np.mean([gini_importance(_tree(gi)) for gi in range(len(ms))], 0)
    with mp.get_context("spawn").Pool(jobs) as pool:
        res = sorted(pool.map(perm_frame_job, list(enumerate(ms))))
    base = np.mean([r[1] for r in res], 0)
    acc = np.array([r[2] for r in res])          # (games, features)
    ll = np.array([r[3] for r in res])
    val = np.array([r[4] for r in res])
    order = np.argsort(-acc.mean(0))
    print("held-out frame accuracy {:.4f}, log loss {:.4f} (fold trees, graded frames, stride 5)".format(*base))
    print("{:22s} {:>7s} {:>11s} {:>9s} {:>10s} {:>9s}".format("feature", "gini", "perm d-acc", "(SE)",
                                                                 "d-logloss", "val-only"))
    top = [int(i) for i in order[:15]] + [int(i) for i in np.argsort(-gini)[:10] if i not in order[:15]]
    for i in top:
        print("{:22s} {:7.3f} {:11.4f} {:9.4f} {:10.4f} {:9.4f}".format(
            names[i], gini[i], acc[:, i].mean(), acc[:, i].std(ddof=1) / np.sqrt(len(ms)), ll[:, i].mean(),
            val[:, i].mean()))
    pos = np.clip(acc.mean(0), 0, None)
    share = pos / pos.sum()
    top3 = np.argsort(-share)[:3]
    print("permutation share of the top 3 ({}) = {:.2f}; gini share of ball_disp_b+lastctrl+any_since_ctrl = {:.2f}"
          .format(", ".join(names[i] for i in top3), share[top3].sum(),
                  sum(gini[names.index(n)] for n in ("ball_disp_b", "lastctrl", "any_since_ctrl"))))
    feats = sorted({names[i] for i in list(order[:8]) + list(np.argsort(-gini)[:6])})
    with mp.get_context("spawn").Pool(jobs) as pool:
        dres = sorted(pool.map(perm_decode_job, [(gi, m, feats) for gi, m in enumerate(ms)]))
    dbase = np.array([r[1]["__base__"] for r in dres])
    print("decode-level (decoder + fold 2 s) ctrl_mean, base {:.4f}; drop when one input is permuted:".format(
        dbase.mean()))
    dec_drop = {}
    for f in feats:
        v = np.array([r[1][f] for r in dres])
        p = paired(dbase, v)
        dec_drop[f] = p
        print("  {:22s} -{:.4f} (SE {:.4f}, {}/7 games)".format(f, p["diff"], p["se"], p["wins"]))
    return dict(names=names, gini=gini.tolist(), perm_acc=acc.tolist(), perm_ll=ll.tolist(),
                perm_val_only=val.tolist(), base=base.tolist(), decode_base=dbase.tolist(), decode_drop=dec_drop)


# ------------------------------------------------------------------ quoted rules, held out
RULES = {
    # name: (README claim: share of rows, purity of the named class)
    "R1 ball missing or disp_b<=0.578 -> D": (0.505, 0.86),
    "R2 R1 & no ctrl >3.02 s & mean_speed_s-0.8<=2.24 & ball_speed<=7.92 -> D": (0.405, 0.98),
    "R3 R1 & ctrl within 3.02 s & ball_out_m2.0>0 -> D": (0.016, 0.83),
    "R4 ball moving & lastctrl team -> that team": (0.495, 0.82),
    "R5 R4 & that team controlled within 3.42 s -> that team": (0.416, 0.88),
    "R6 R5 & its nearest player closer by >1.25 m (2 s mean) -> that team": (0.282, 0.94),
}


def rules_job(args):
    gi, match = args
    setup()
    import bench
    pm = bench.pm
    out = {k: [0, 0, 0] for k in RULES}   # covered, correct, graded
    for seed in SEEDS:
        _, _, gt, _, F, names, _ = _game_data(match, seed)
        g = gt != pm.X
        c = {n: F[:, i].astype(np.float64) for i, n in enumerate(names)}
        disp = c["ball_disp_b"]
        r1 = np.isnan(disp) | (disp <= 0.578)
        recent = c["any_since_ctrl"] <= 4.34          # NaN -> right (False)
        ms_ = c["mean_speed_s-0.8"]
        bs = c["ball_speed"]
        r2 = r1 & ~recent & (ms_ <= 2.24) & (np.isnan(bs) | (bs <= 7.92))
        r3 = r1 & recent & (c["ball_out_m2.0"] > 5.43e-5)
        moving = ~r1
        lc = c["lastctrl"]
        team_lab = np.where(lc > 0.5, pm.H, np.where(lc < -0.5, pm.A, -1))
        since = np.where(lc > 0.5, c["H_since_ctrl"], c["A_since_ctrl"])
        r4 = moving & (team_lab >= 0)
        r5 = r4 & (np.isnan(since) | (since <= 4.46))
        dd = np.where(lc > 0.5, -c["diff_d1_m2.0"], c["diff_d1_m2.0"])   # > 0: that team closer
        r6 = r5 & (dd > 1.25)
        for k, m, lab in (("R1 ball missing or disp_b<=0.578 -> D", r1, pm.D),
                          ("R2 R1 & no ctrl >3.02 s & mean_speed_s-0.8<=2.24 & ball_speed<=7.92 -> D", r2, pm.D),
                          ("R3 R1 & ctrl within 3.02 s & ball_out_m2.0>0 -> D", r3, pm.D),
                          ("R4 ball moving & lastctrl team -> that team", r4, team_lab),
                          ("R5 R4 & that team controlled within 3.42 s -> that team", r5, team_lab),
                          ("R6 R5 & its nearest player closer by >1.25 m (2 s mean) -> that team", r6, team_lab)):
            m = m & g
            out[k][0] += int(m.sum())
            out[k][1] += int((gt[m] == (lab if np.isscalar(lab) else lab[m])).sum())
            out[k][2] += int(g.sum())
    return gi, out


def part_rules(jobs):
    ms = matches()
    with mp.get_context("spawn").Pool(jobs) as pool:
        res = sorted(pool.map(rules_job, list(enumerate(ms))))
    out = {}
    print("{:74s} {:>14s} {:>22s}".format("rule (README thresholds)", "claim cov/pur", "held-out cov/pur [min-max game]"))
    for k, (cov, pur) in RULES.items():
        covs = np.array([r[1][k][0] / r[1][k][2] for r in res])
        purs = np.array([r[1][k][1] / max(r[1][k][0], 1) for r in res])
        tot = sum(r[1][k][1] for r in res) / sum(r[1][k][0] for r in res)
        out[k] = dict(claim=(cov, pur), coverage=float(covs.mean()), purity=float(tot),
                      purity_min=float(purs.min()), purity_max=float(purs.max()))
        print("{:74s} {:5.3f} / {:4.2f}   {:5.3f} / {:4.3f} [{:.3f}-{:.3f}]".format(
            k, cov, pur, covs.mean(), tot, purs.min(), purs.max()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", default="all")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "audit.json")
    res = json.load(open(path)) if os.path.isfile(path) else {}
    parts = ["control", "ensemble", "rules", "perm"] if a.part == "all" else [a.part]
    for p in parts:
        print("== " + p, flush=True)
        res[p] = {"control": part_control, "ensemble": part_ensemble}[p]() if p in ("control", "ensemble") \
            else {"perm": part_perm, "rules": part_rules}[p](a.jobs)
        with open(path, "w") as handle:
            json.dump(res, handle)


if __name__ == "__main__":
    main()
