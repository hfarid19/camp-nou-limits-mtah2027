#!/usr/bin/env python3
"""bench.py --dfl (exactly, imported) + dead-ball metrics of the decoded output, per game.

    python candidates/ballfree_dead/dead_report.py candidates/ballfree_dead/pipeline.py --folds F

Dead-ball metrics (the ground truth's D = DFL BallStatus 0):
  deadF1    frame-level F1 of D vs alive (graded frames)
  stopF1    stoppage-level F1 after Lang et al. (2022): maximal D runs >= 2 s, one-to-one greedy
            matching at temporal IoU >= 0.5
  startMAE / endMAE   mean |offset| (s) of matched stoppage starts / ends
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench  # noqa: E402

CAP = []
_orig = bench.pm.evaluate


def _ev(gt, pred, **kw):
    CAP.append((np.asarray(gt).copy(), np.asarray(pred).copy()))
    return _orig(gt, pred, **kw)


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def stoppage_f1(gt, pred, fps=25, min_s=2.0, thr=0.5):
    graded = gt != 3
    g = [r for r in runs((gt == 2) & graded) if r[1] - r[0] >= min_s * fps]
    p = [r for r in runs((pred == 2) & graded) if r[1] - r[0] >= min_s * fps]
    cand = []
    for i, (a, b) in enumerate(g):
        for j, (c, d) in enumerate(p):
            inter = min(b, d) - max(a, c)
            if inter > 0:
                iou = inter / (max(b, d) - min(a, c))
                if iou >= thr:
                    cand.append((iou, i, j))
    cand.sort(reverse=True)
    ug, up, m = set(), set(), []
    for iou, i, j in cand:
        if i not in ug and j not in up:
            ug.add(i)
            up.add(j)
            m.append((i, j))
    tp = len(m)
    prec = tp / max(len(p), 1)
    rec = tp / max(len(g), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    s_err = np.mean([abs(g[i][0] - p[j][0]) / fps for i, j in m]) if m else np.nan
    e_err = np.mean([abs(g[i][1] - p[j][1]) / fps for i, j in m]) if m else np.nan
    return f1, prec, rec, len(g), len(p), s_err, e_err


def dead_metrics(gt, pred):
    graded = gt != 3
    t, q = gt[graded] == 2, pred[graded] == 2
    tp = (t & q).sum()
    f1 = 2 * tp / max(t.sum() + q.sum(), 1)
    sf = stoppage_f1(gt, pred)
    return dict(deadF1=round(float(f1), 4), stopF1=round(sf[0], 4), stopP=round(sf[1], 4),
                stopR=round(sf[2], 4), n_stop=f"{sf[4]}/{sf[3]}", startMAE=round(float(sf[5]), 2),
                endMAE=round(float(sf[6]), 2), pred_dead_share=round(float(q.mean()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--folds", required=True)
    ap.add_argument("--save-probs", default=None)
    a = ap.parse_args()
    bench.pm.evaluate = _ev
    folds = json.load(open(a.folds))
    rows = bench.bench_dfl(a.candidate, folds, a.folds, a.save_probs)
    out = []
    for r, (gt, pred) in zip(rows, CAP):
        dm = dead_metrics(gt, pred)
        out.append(dict(r, **dm))
        print("  {game} spellF1 {spellF1:.4f} macroF1 {macroF1:.4f} poss {poss} | deadF1 {deadF1:.4f}"
              " stopF1 {stopF1:.4f} (P {stopP:.3f} R {stopR:.3f} n {n_stop}) start/end MAE "
              "{startMAE:.2f}/{endMAE:.2f}s dead share {pred_dead_share}".format(**out[-1]))
    keys = ["spellF1", "macroF1", "deadF1", "stopF1", "startMAE", "endMAE"]
    mean = {k: round(float(np.mean([o[k] for o in out])), 4) for k in keys}
    print("MEAN " + json.dumps(mean))


if __name__ == "__main__":
    main()
