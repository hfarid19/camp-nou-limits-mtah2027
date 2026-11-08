"""Round-1 sweep for hier_touch_decoder (dev bench rows only: Sample Games 1+2 x seeds 2027/2028,
scored with possession_metrics.evaluate exactly as candidates/bench.py does).

Two views of every config: 'clean' (the bench rows as-is) and 'off' (the same rows with an AR(1)
ball offset, sd 1.3 m, tau 1 s, seed 7 -- the robustness check the judges used). Features and the
learned_dfl_evidence frame probabilities are cached per row/view in tuning/_cache (not committed).

usage: python tuning/sweep_r1.py '[{"w_lde": 2.0, "w_shape": 0.0}, ...]'
"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAND))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench  # noqa: E402
pm, tsd = bench.pm, bench.tsd
htd = bench.load_candidate(os.path.join(CAND, "pipeline.py"))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)


def offset(ball, rng, sd=1.3, tau=25):
    n = len(ball); a = np.exp(-1.0 / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), (n, 2)); off = np.zeros((n, 2))
    for t in range(1, n):
        off[t] = a * off[t - 1] + e[t]
    return (ball + off).astype(np.float32)


def rows(view):
    out = []
    for name in bench.DEV:
        for seed in bench.SEEDS:
            path = os.path.join(CACHE, "%s_%s_%d.pkl" % (view, name, seed))
            slots, st, ball, gt, blocks, sign = bench.prepared(name, seed)
            if view == "off":
                ball = offset(ball.copy(), np.random.default_rng(7))
            if os.path.isfile(path):
                with open(path, "rb") as h:
                    Fe = pickle.load(h)
            else:
                Fe = htd.features(slots, st, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, 25)
                Fe = htd.add_lde(Fe, slots, st, ball, sign, blocks, tsd.PITCH, 25,
                                 dict(htd.CFG, w_lde=1.0))
                Fe.pop("v1", None)
                with open(path, "wb") as h:
                    pickle.dump(Fe, h)
            out.append((gt, blocks, Fe))
    return out


R = {v: rows(v) for v in ("clean", "off")}


def score(over):
    C = dict(htd.CFG, **over)
    res = {}
    for v, rr in R.items():
        sc = []
        for gt, blocks, Fe in rr:
            Fe = dict(Fe)
            if "feat" in over:
                Fe = htd.features_override(Fe, over["feat"]) if hasattr(htd, "features_override") else Fe
            p = htd.decode(Fe, blocks, C)
            r = pm.evaluate(gt, p, bootstrap_n=0)
            sc.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
        a = np.array(sc)
        res[v] = (round(a[:, 0].mean(), 4), round(a[:, 1].mean(), 4), int(a[:, 2].sum()),
                  [round(x, 4) for x in a[:, 0]])
    return res


if __name__ == "__main__":
    for over in json.loads(sys.argv[1]):
        r = score(over)
        print(json.dumps(over), "clean %.4f/%.4f n%d" % r["clean"][:3], "off %.4f/%.4f n%d" % r["off"][:3],
              "mean %.4f" % ((r["clean"][0] + r["off"][0]) / 2), r["clean"][3], flush=True)
