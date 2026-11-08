"""Round-2 sweep for hier_touch_decoder (dev bench rows only: Sample Games 1+2 x seeds 2027/2028,
scored with possession_metrics.evaluate exactly as candidates/bench.py does).

Four views of every config (all fixed seeds, deterministic):
  clean  the bench rows as-is
  off    AR(1) ball offset, sd 1.3 m, tau 1 s, seed 7 (round-1 check; in-distribution for the
         learned evidence, which trains on the same degradation)
  s2     out-of-distribution stress: AR(1) offset sd 2 m, tau 3 s, plus extra 0.5-2 s ball
         dropouts (seed 11) -- the measured-performance judge's 'stress2'
  osd    offset then 6-frame smoothing of the ball, plus chunked player dropout (seeds 7/8) --
         the judges' 'osd' wrapper
Features and the frozen lde_frozen/ probabilities are cached per row/view in tuning/_cache_r2
(git-ignored; the round-1 _cache held probabilities of models that no longer exist).

usage: python tuning/sweep_r2.py '[{"w_shape": 0.0}, ...]' [views, default clean,off,s2,osd]
"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAND))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench  # noqa: E402
pm, tsd, pp = bench.pm, bench.tsd, None
htd = bench.load_candidate(os.path.join(CAND, "pipeline.py"))
pp = htd.pp
CACHE = os.path.join(HERE, "_cache_r2"); os.makedirs(CACHE, exist_ok=True)


def offset(ball, rng, sd=1.3, tau=25, init=False):
    n = len(ball); a = np.exp(-1.0 / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), (n, 2)); off = np.zeros((n, 2))
    if init:
        off[0] = rng.normal(0, sd, 2)
    for t in range(1, n):
        off[t] = a * off[t - 1] + e[t]
    return (ball + off).astype(np.float32)


def smooth(ball, hw):
    seen = ~np.isnan(ball[:, 0])
    out = np.stack([pp.moving_mean(ball[:, i].astype(np.float64), hw) for i in (0, 1)], 1)
    out[~seen] = np.nan
    return out.astype(np.float32)


def drop(players, rng, rate=0.06, mean_len=50):
    players = players.copy(); n, S, _ = players.shape
    starts = rng.random((n, S)) < rate / mean_len
    for s, j in zip(*np.where(starts)):
        players[s:s + int(rng.exponential(mean_len)) + 1, j] = np.nan
    return players


def view_input(view, slots, ball):
    if view == "off":
        return slots, offset(ball.copy(), np.random.default_rng(7))
    if view == "s2":
        rng = np.random.default_rng(11)
        b = offset(ball.copy(), rng, sd=2.0, tau=75)
        for t in np.flatnonzero(rng.random(len(b)) < 0.2 / 30):
            b[t:t + int(rng.integers(12, 50))] = np.nan
        return slots, b
    if view == "osd":
        b = smooth(offset(ball.copy(), np.random.default_rng(7), init=True), 6)
        return drop(slots, np.random.default_rng(8)), b
    return slots, ball


def rows(view):
    out = []
    for name in bench.DEV:
        for seed in bench.SEEDS:
            path = os.path.join(CACHE, "%s_%s_%d.pkl" % (view, name, seed))
            slots, st, ball, gt, blocks, sign = bench.prepared(name, seed)
            if os.path.isfile(path):
                with open(path, "rb") as h:
                    Fe = pickle.load(h)
            else:
                slots, ball = view_input(view, slots, ball)
                Fe = htd.features(slots, st, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, 25)
                Fe = htd.add_lde(Fe, slots, st, ball, sign, blocks, tsd.PITCH, 25,
                                 dict(htd.CFG, w_lde=1.0))
                Fe.pop("v1", None)
                with open(path, "wb") as h:
                    pickle.dump(Fe, h)
            out.append((gt, blocks, Fe))
    return out


VIEWS = (sys.argv[2] if len(sys.argv) > 2 else "clean,off,s2,osd").split(",")
R = {v: rows(v) for v in VIEWS}


def score(over):
    C = dict(htd.CFG, **over)
    res = {}
    for v, rr in R.items():
        sc = []
        for gt, blocks, Fe in rr:
            p = htd.decode(dict(Fe), blocks, C)
            r = pm.evaluate(gt, p, bootstrap_n=0)
            sc.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
        a = np.array(sc)
        res[v] = (round(a[:, 0].mean(), 4), round(a[:, 1].mean(), 4), int(a[:, 2].sum()),
                  [round(x, 4) for x in a[:, 0]])
    return res


if __name__ == "__main__":
    for over in json.loads(sys.argv[1]):
        r = score(over)
        txt = " | ".join("%s %.4f" % (v, r[v][0]) for v in VIEWS)
        print("%-60s %s | macro %.4f n%d %s" % (json.dumps(over), txt, r[VIEWS[0]][1],
                                                 r[VIEWS[0]][2], r[VIEWS[0]][3]), flush=True)
