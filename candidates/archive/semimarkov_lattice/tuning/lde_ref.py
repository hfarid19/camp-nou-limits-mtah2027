import sys, json, numpy as np, devrun
from multiprocessing import Pool
import possession_metrics as pm
def job(a):
    m, cfg = a
    import pipeline as P
    L = P.L
    ev, gt, blocks = devrun.load(devrun.EV + "/%s.npz" % m)
    names = ["ball_seen", "ball_out", "H_d1", "A_d1"]
    F = np.stack([ev["raw_seen"].astype(float), ev["out"].astype(float), ev["d_home"], ev["d_away"]], 1)
    aux = dict(F=F, names=names, p_change=ev["pc"], change_prior=float(ev["change_prior"]), long_gap=None)
    pred = L.decode(1 / (1 + np.exp(-ev["ld"])), 1 / (1 + np.exp(-ev["l6"])), aux, blocks, len(gt), 25, cfg=cfg)
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return r["spell_f1_mean"], r["n_pred_spells"]
if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    with Pool(2) as p:
        res = p.map(job, [(m, cfg) for m in ["dev_Sample_Game_1_2027","dev_Sample_Game_1_2028","dev_Sample_Game_2_2027","dev_Sample_Game_2_2028"]])
    print(json.dumps(cfg), "LDE-decoder spellF1 %.4f poss %d" % (np.mean([r[0] for r in res]), sum(r[1] for r in res)))
