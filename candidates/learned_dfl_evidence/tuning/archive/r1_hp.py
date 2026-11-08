import numpy as np, multiprocessing as mp, sys, json
import r1_ev as ev
sys.path.insert(0, ev.ROOT); import possession_pipeline as pp
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def mk(k, hw_s, sw=None):
    def fn(r, aux, c):
        l = lg(r["ph"])
        lh = l + k * (l - pp.moving_mean(l, int(hw_s * 25)))
        ph = 1 / (1 + np.exp(-lh))
        cfg = {} if sw is None else dict(switch_cost=sw)
        return ev.L.decode(r["pdd"], ph, aux, r["blocks"], len(r["ball"]), 25, cfg=cfg)
    return fn
MODES = ["clean_sm", "osd_sm", "offset_sm"]
def one(a):
    return a, [ev.score(m, fn=mk(*a)) for m in MODES]
if __name__ == "__main__":
    for m in MODES: ev.rows(m)
    grid = json.loads(sys.argv[1])
    with mp.get_context("fork").Pool(8) as p:
        for a, r in p.imap(one, [tuple(g) for g in grid]):
            print(a, [x[:3] for x in r], round(np.mean([x[0] for x in r]), 4), flush=True)
