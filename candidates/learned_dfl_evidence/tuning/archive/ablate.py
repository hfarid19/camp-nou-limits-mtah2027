import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench, possession_metrics as pm, possession_pipeline as pp
sys.path.insert(0, os.path.join(ROOT, "candidates", "learned_dfl_evidence"))
import pipeline as lp
res = {}
for name in bench.DEV:
    for seed in bench.SEEDS:
        slots, st, ball, gt, blocks, sign = bench.prepared(name, seed)
        pdl, phl, aux = lp.frame_probs(slots, st, ball, sign, blocks, (105., 68.), 25)
        E, _ = pp.build_evidence(slots, ball, st, sign, blocks, [], ("Home", "Away"), (105., 68.), dict(pp.PARAMS, temperature=1.0), lambda *a, **k: None)
        pdv = np.exp(E[:, 2]); phv = 1 / (1 + np.exp(E[:, 1] - E[:, 0]))
        for lab, pd_, ph in [("learned both", pdl, phl), ("learned dead + v1 team", pdl, phv), ("v1 dead + learned team", pdv, phl), ("v1 both (new decoder)", pdv, phv)]:
            for sc in (0.75, 2.25):
                r = pm.evaluate(gt, lp.decode(pd_, ph, aux, blocks, len(gt), 25, cfg=dict(switch_cost=sc)), bootstrap_n=0)
                res.setdefault((lab, sc), []).append((r["spell_f1_mean"], r["macro_f1"]))
for k, v in res.items():
    a = np.array(v); print(k, a.mean(0).round(4), a[:, 0].round(3))
