import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench, possession_metrics as pm
sys.path.insert(0, os.path.join(ROOT, "candidates", "learned_dfl_evidence"))
import pipeline as lp, possession_pipeline as pp
for name in bench.DEV[:2]:
    slots, st, ball, gt, blocks, sign = bench.prepared(name, 2027)
    pd_, ph, aux = lp.frame_probs(slots, st, ball, sign, blocks, (105., 68.), 25)
    g = gt < 3
    alive = gt < 2
    print(name, "dead acc", ((pd_ > 0.5) == (gt == 2))[g].mean().round(3), "HA acc", ((ph > 0.5) == (gt == 0))[alive].mean().round(3),
          "pred dead share", (pd_[g] > 0.5).mean().round(3), "gt dead", (gt[g] == 2).mean().round(3))
    segs = pp.segments(np.where(g, gt, 2))
    poss = [(s, e, l) for s, e, l in segs if l < 2]
    du = np.array([e - s for s, e, l in poss]) / 25
    agree = np.array([((ph[s:e] > 0.5) == (l == 0)).mean() for s, e, l in poss])
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]:
        m = (du >= lo) & (du < hi)
        print("  gt spells {}-{}s: n {} mean frame agree {:.2f}".format(lo, hi, m.sum(), agree[m].mean()))
    pred = lp.decode(pd_, ph, aux, blocks, len(gt), 25, cfg=dict(switch_cost=0.5))
    ps = [(s, e, l) for s, e, l in pp.segments(np.where(g, pred, 2)) if l < 2]
    pdu = np.array([e - s for s, e, l in ps]) / 25
    print("  pred dur pct", np.percentile(pdu, [10, 25, 50, 75, 90]).round(1), "gt", np.percentile(du, [10, 25, 50, 75, 90]).round(1))
    pdd = [(s, e) for s, e, l in pp.segments(np.where(g, pred, 3)) if l == 2]
    gdd = [(s, e) for s, e, l in segs if l == 2]
    print("  D segs pred", len(pdd), "gt", len(gdd), "pred dead share", (pred[g] == 2).mean().round(3))
