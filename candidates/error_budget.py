#!/usr/bin/env python3
"""Where do a candidate's possession errors come from? Error budget + oracle headroom on DFL.

    python candidates/error_budget.py <codes dir> [--split dfl2]

<codes dir> holds <split>_<match>_<seed>_codes.npy as written by
`bench.py ... --dfl2 --save-probs DIR`. Ground truth, ball and blocks come from the bench's DFL
caches, so this only ever reads the open DFL labels (never the competition's).

Part 1 classifies every ground-truth possession spell missed at IoU 0.5 (and every spurious
predicted one) by duration, ball visibility, neighbours and what the prediction did there.
Part 2 replaces one kind of error with the truth and rescores (oracles), which bounds how much
spellF1 fixing that error class could buy.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
sys.path.insert(0, HERE)
import bench                     # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
DUR_BUCKETS = [(0, 1), (1, 3), (3, 8), (8, 1e9)]
IOU = pm.SPELL_IOU_HEADLINE


def bucket(sec):
    for lo, hi in DUR_BUCKETS:
        if lo <= sec < hi:
            return "{}-{}s".format(lo, hi if hi < 1e8 else "")
    return "?"


def spell_f1(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def neighbours(segs, i):
    prev = segs[i - 1][2] if i > 0 else None
    nxt = segs[i + 1][2] if i + 1 < len(segs) else None
    kind = lambda c: "start" if c is None else ("D" if c == pm.D else "team")   # noqa: E731
    return "{}|{}".format(kind(prev), kind(nxt))


def analyse(gt, pred, ball):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    g_sp, p_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(g_sp, p_sp)
    flags = pm.matched_gt_flags(ov, len(g_sp), IOU)
    matched_pred, used = set(), set()
    for iou, gi, pj in ov:                                       # the same greedy match, pred side
        if iou < IOU:
            break
        if gi in used or pj in matched_pred:
            continue
        used.add(gi)
        matched_pred.add(pj)
    segs = pm.segments(gt, runs)
    seg_index = {(a, b): i for i, (a, b, _) in enumerate(segs)}
    seen = ~np.isnan(ball[:, 0])
    missed, spurious = [], []
    for k, (a, b, c) in enumerate(g_sp):
        if flags[k]:
            continue
        lab = np.bincount(filled[a:b], minlength=4)
        other = pm.A if c == pm.H else pm.H
        best_same = max([iou for iou, gi, _ in ov if gi == k] or [0.0])
        if lab[pm.D] >= max(lab[c], lab[other]):
            why = "predicted dead"
        elif lab[other] > lab[c]:
            why = "wrong team"
        elif best_same > 0 and any(pa <= a and pb >= b for pa, pb, pc in p_sp if pc == c):
            why = "merged into longer same-team spell"
        else:
            why = "boundary off (IoU<0.5)"
        missed.append(dict(dur=(b - a) / FPS, why=why, vis=float(seen[a:b].mean()),
                           nb=neighbours(segs, seg_index[(a, b)])))
    for j, (a, b, c) in enumerate(p_sp):
        if j in matched_pred:
            continue
        lab = np.bincount(gt[a:b], minlength=4)
        other = pm.A if c == pm.H else pm.H
        if lab[pm.D] >= max(lab[c], lab[other]):
            why = "over dead ball"
        elif lab[other] > lab[c]:
            why = "wrong team"
        else:
            why = "fragment of a true spell"
        spurious.append(dict(dur=(b - a) / FPS, why=why))
    return dict(n_gt=len(g_sp), n_pred=len(p_sp), missed=missed, spurious=spurious,
                gt_durs=[(b - a) / FPS for a, b, _ in g_sp],
                gt_vis=[float(seen[a:b].mean()) for a, b, _ in g_sp], flags=flags)


def oracles(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    out = {"as predicted": spell_f1(gt, filled)}
    # (a) true segmentation, predicted labels: majority prediction inside every true segment
    o = filled.copy()
    for a, b, _ in pm.segments(gt, runs):
        o[a:b] = np.argmax(np.bincount(filled[a:b], minlength=4)[:3])
    out["true boundaries, pred labels"] = spell_f1(gt, o)
    # (b) predicted segmentation, true labels: majority truth inside every predicted segment
    o = filled.copy()
    for a, b, _ in pm.segments(filled, runs):
        o[a:b] = np.argmax(np.bincount(gt[a:b], minlength=4)[:3])
    out["pred boundaries, true labels"] = spell_f1(gt, o)
    # (c) perfect dead ball: D exactly where the truth has D; elsewhere the last predicted team
    o = filled.copy()
    o[gt == pm.D] = pm.D
    fix = (filled == pm.D) & (gt != pm.D)
    team = np.where(filled == pm.D, -1, filled)
    for a, b in runs:
        last = -1
        for t in range(a, b):
            if team[t] >= 0:
                last = team[t]
            elif fix[t]:
                o[t] = last if last >= 0 else gt[t]
    out["perfect dead ball"] = spell_f1(gt, o)
    # (d) short possessions inserted perfectly
    for lim in (1.0, 2.0):
        o = filled.copy()
        for a, b, c in pm.possession_spells(gt, runs):
            if (b - a) / FPS < lim:
                o[a:b] = c
        out["true spells < {:.0f}s pasted in".format(lim)] = spell_f1(gt, o)
    # (e) every true H<->A change placed exactly, predicted labels otherwise (splits only)
    o = filled.copy()
    for a, b, c in pm.possession_spells(gt, runs):
        if np.bincount(filled[a:b], minlength=4)[c] > 0:
            o[a:b] = np.where(filled[a:b] == pm.D, pm.D, c)
    out["true team inside every true spell (keep pred D)"] = spell_f1(gt, o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes_dir")
    ap.add_argument("--split", default="dfl2")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.codes_dir, args.split + "_*_codes.npy")))
    assert files, "no {}_*_codes.npy in {}".format(args.split, args.codes_dir)
    prep = bench.dfl2_prepared if args.split == "dfl2" else bench.dfl_prepared
    res, orc = [], []
    for f in files:
        match, seed = os.path.basename(f)[len(args.split) + 1:-len("_codes.npy")].rsplit("_", 1)
        _, _, ball, gt, _, _ = prep(match, int(seed))
        pred = np.load(f)
        res.append(analyse(gt, pred, ball))
        orc.append(oracles(gt, pred))
    n_gt = sum(r["n_gt"] for r in res)
    n_pred = sum(r["n_pred"] for r in res)
    missed = [m for r in res for m in r["missed"]]
    spur = [s for r in res for s in r["spurious"]]
    print("{} games x seeds: {} true spells, {} predicted, {} missed at IoU {}, {} spurious".format(
        len(files), n_gt, n_pred, len(missed), IOU, len(spur)))

    def tab(title, items, key, total=None):
        print("\n" + title)
        keys = sorted({key(i) for i in items})
        for k in keys:
            c = sum(1 for i in items if key(i) == k)
            print("  {:45s} {:5d}  {:5.1%}".format(str(k), c, c / max(len(items), 1)))
    tab("MISSED true spells by reason", missed, lambda m: m["why"])
    tab("MISSED by duration", missed, lambda m: bucket(m["dur"]))
    tab("MISSED by reason x duration", missed, lambda m: "{} / {}".format(m["why"], bucket(m["dur"])))
    tab("MISSED by neighbours (prev|next in truth)", missed, lambda m: m["nb"])
    tab("MISSED by ball visible during the spell", missed,
        lambda m: "visible {}".format("all" if m["vis"] == 1 else ("none" if m["vis"] == 0 else "part")))
    tab("SPURIOUS predicted spells by reason", spur, lambda s: s["why"])
    tab("SPURIOUS by duration", spur, lambda s: bucket(s["dur"]))

    print("\nRECALL by true-spell duration (all games)")
    durs = np.concatenate([r["gt_durs"] for r in res])
    hits = np.concatenate([r["flags"] for r in res])
    for lo, hi in DUR_BUCKETS:
        sel = (durs >= lo) & (durs < hi)
        print("  {:8s} n {:5d} ({:4.1%} of true spells)  recall {:.3f}".format(
            bucket(lo), int(sel.sum()), sel.mean(), hits[sel].mean() if sel.any() else 0))
    vis = np.concatenate([r["gt_vis"] for r in res])
    for name, sel in (("ball always visible", vis == 1), ("ball partly missing", (vis > 0) & (vis < 1)),
                      ("ball never visible", vis == 0)):
        print("  {:22s} n {:5d}  recall {:.3f}".format(name, int(sel.sum()), hits[sel].mean() if sel.any() else 0))

    print("\nORACLES (mean spellF1 over games x seeds; headroom vs as predicted)")
    base = np.mean([o["as predicted"] for o in orc])
    for k in orc[0]:
        v = np.mean([o[k] for o in orc])
        print("  {:50s} {:.4f}  {:+.4f}".format(k, v, v - base))


if __name__ == "__main__":
    main()
