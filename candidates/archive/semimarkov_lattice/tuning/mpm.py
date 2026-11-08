import sys, json, numpy as np, devrun, time
from multiprocessing import Pool
import possession_metrics as pm
def job(a):
    gi, m, cfg = a
    import pipeline as P
    ev, gt, blocks = devrun.load(devrun.EV + "/dfl_%s.npz" % m)
    pred, proba = P.decode_ev(ev, blocks, len(gt), 25, devrun.tab_for(gi), cfg, True)
    g = gt < 3
    ll = -np.mean(np.log(np.clip(proba[g, gt[g]], 1e-6, 1)))
    acc = (proba[g].argmax(1) == gt[g]).mean()
    mp = pred.copy()
    for _, f, l in blocks:
        mp[f:l+1] = proba[f:l+1].argmax(1)
    r1 = pm.evaluate(gt, pred, bootstrap_n=0); r2 = pm.evaluate(gt, mp, bootstrap_n=0)
    return m, r1["spell_f1_mean"], r2["spell_f1_mean"], r2["n_pred_spells"], ll, acc
if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]); t = time.time()
    with Pool(2) as p:
        res = p.map(job, [(gi, m, cfg) for gi, m in enumerate(devrun.MATCHES)])
    for r in res: print("  %s map %.4f mpm %.4f (%d) logloss %.4f acc %.4f" % r)
    print(json.dumps(cfg), "MAP %.4f MPM %.4f logloss %.4f acc %.4f  %.0fs" % tuple([np.mean([r[k] for r in res]) for k in (1, 2, 4, 5)] + [time.time() - t]))
