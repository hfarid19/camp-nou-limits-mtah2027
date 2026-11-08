import numpy as np, multiprocessing as mp
import r1_ev as ev
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def mk(bd, bh=0.0):
    def fn(r, aux, c):
        pdd = 1 / (1 + np.exp(-(lg(r["pdd"]) + bd)))
        return ev.L.decode(pdd, r["ph"], aux, r["blocks"], len(r["ball"]), 25)
    return fn
def one(bd):
    return bd, [ev.score(m, fn=mk(bd)) for m in ["clean_sm", "osd_sm", "offset_sm"]]
if __name__ == "__main__":
    for m in ["clean_sm", "osd_sm", "offset_sm"]: ev.rows(m)
    with mp.get_context("fork").Pool(6) as p:
        for bd, r in p.imap(one, [-1.0, -0.5, -0.21, 0.21, 0.5, 1.0]):
            print(bd, [x[:3] for x in r], round(np.mean([x[0] for x in r]), 4), flush=True)
