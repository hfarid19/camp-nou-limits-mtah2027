import sys, json, pickle, os, numpy as np
import r1_ev as ev
HERE = ev.HERE
H = {m: pickle.load(open(os.path.join(HERE, "htd_%s.pkl" % m), "rb")) for m in ["clean", "osd", "offset"]}
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def mk(mode, a, wgap, hard=False):
    it = iter(H[mode])
    def fn(r, aux, c):
        h = next(it)
        if hard:
            pdd = np.where(h["mask"], 0.999, 0.001)
            aux = dict(aux, long_gap=None)
        else:
            ldd = (1 - a) * lg(r["pdd"]) + a * h["dl"]
            pdd = 1 / (1 + np.exp(-ldd))
        return ev.L.decode(pdd, r["ph"], aux, r["blocks"], len(r["ball"]), 25, cfg=dict(w_gap=wgap))
    return fn
for e in json.loads(sys.argv[1]):
    res = {m: ev.score(m, fn=mk(m, e["a"], e["wgap"], e.get("hard", False))) for m in ["clean", "osd", "offset"]}
    print(e, res, flush=True)
