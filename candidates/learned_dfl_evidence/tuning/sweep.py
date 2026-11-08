"""Decoder sweep on the dev bench (Games 1+2 x seeds 2027/2028), probabilities computed once."""
import os
import itertools, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench
import possession_metrics as pm
sys.path.insert(0, os.path.join(ROOT, "candidates", "learned_dfl_evidence"))
import pipeline as lp

cache = []
for name in bench.DEV:
    for seed in bench.SEEDS:
        slots, st, ball, gt, blocks, sign = bench.prepared(name, seed)
        pd_, ph, aux = lp.frame_probs(slots, st, ball, sign, blocks, (105., 68.), 25)
        shape = lp.v1_shape_logit(slots, st, ball, sign, blocks, [], ("Home", "Away"), (105., 68.))
        cache.append((name, seed, gt, blocks, pd_, ph, aux, shape))

def score(cfg):
    rows = []
    for name, seed, gt, blocks, pd_, ph, aux, shape in cache:
        pred = lp.decode(pd_, ph, aux, blocks, len(gt), 25, cfg=cfg, shape_logit=shape)
        r = pm.evaluate(gt, pred, bootstrap_n=0)
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
    a = np.array(rows)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 0], a[:, 2]

grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
keys = list(grid)
res = []
for combo in itertools.product(*[grid[k] for k in keys]):
    cfg = dict(zip(keys, combo))
    s, m, per, npred = score(cfg)
    res.append((s, m, cfg, per, npred))
    print("{:.4f} {:.4f} {} per {} poss {}".format(s, m, cfg, np.round(per, 3), npred.astype(int)), flush=True)
print("BEST")
for r in sorted(res, key=lambda r: -r[0])[:8]:
    print("{:.4f} {:.4f} {}".format(r[0], r[1], r[2]))
