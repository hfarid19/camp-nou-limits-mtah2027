"""Round 3 tuning harness: cache frame probabilities once (dev 4 views + DFL leave-one-game-out
fold models), then score decode() variants on all five views. Dev = bench rows (Sample Games 1+2
x 2 seeds), views clean / offset / osd / stress2 (the judges' stress views, same seeds); DFL =
bench --dfl rows with the fold models named in $LDE_FOLDS (bench --folds JSON). Never touches
evaluation/gt, the holdout game or the match.
usage: python tuning/r3_views.py '[{cfg}, {cfg}, ...]'   (cache in $LDE_TUNE_CACHE, else external/tune)"""
import os, sys, json, pickle, importlib.util
import numpy as np
C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.environ.get("LDE_TUNE_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "tune"))
os.makedirs(HERE, exist_ok=True)
sys.path.insert(0, C)
import bench
pm, pp, tsd = bench.pm, bench.pp, bench.tsd
sys.path.insert(0, os.path.join(C, "learned_dfl_evidence"))
import pipeline as lp
KEEP = ["H_d1", "A_d1", "ball_seen", "ball_out"]

def _offset(ball, rng, sd=1.3, tau=25):
    n = len(ball); a = np.exp(-1.0 / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), (n, 2)); off = np.zeros((n, 2)); off[0] = rng.normal(0, sd, 2)
    for t in range(1, n):
        off[t] = a * off[t - 1] + e[t]
    return (ball + off).astype(np.float32)
def _smooth(ball, hw=4):
    seen = ~np.isnan(ball[:, 0])
    out = np.stack([pp.moving_mean(ball[:, i].astype(np.float64), hw) for i in (0, 1)], 1); out[~seen] = np.nan
    return out.astype(np.float32)
def _drop(players, rng, rate=0.06, mean_len=50):
    players = players.copy()
    starts = rng.random(players.shape[:2]) < rate / mean_len
    for s, j in zip(*np.where(starts)):
        players[s:s + int(rng.exponential(mean_len)) + 1, j] = np.nan
    return players
def view(name, players, ball):
    """Judges' views (same seeds as v2_stacked_fixes/tuning/views.py)."""
    if name == "clean": return players, ball
    if name == "offset": return players, _offset(ball.copy(), np.random.default_rng(7))
    if name == "osd": return _drop(players, np.random.default_rng(8)), _offset(_smooth(ball), np.random.default_rng(7))
    rng = np.random.default_rng(11); b = _offset(ball.copy(), rng, sd=2.0, tau=75)
    for t in np.flatnonzero(rng.random(len(b)) < 0.2 / 30):
        b[t:t + int(rng.integers(12, 50))] = np.nan
    return players, b

def slim(pd_, ph, aux, players, ball):
    col = {k: i for i, k in enumerate(aux["names"])}
    a = dict(F=aux["F"][:, [col[k] for k in KEEP]].copy(), names=KEEP, p_change=aux["p_change"],
             change_prior=aux["change_prior"], kick_restarts={})
    a["long_gap"] = lp.long_gap_gated(players, ball, 25, lp.CFG["gap_s"], lp.CFG["gap_speed_ms"])
    return (pd_, ph, a)

def cache_dev():
    path = os.path.join(HERE, "cache_dev.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = {}
    for v in ("clean", "offset", "osd", "stress2"):
        rows = []
        for g in bench.DEV:
            for seed in bench.SEEDS:
                slots, st, ball, gt, blocks, sign = bench.prepared(g, seed)
                pl, b = view(v, slots, ball)
                pd_, ph, aux = lp.frame_probs(pl, st, b, sign, blocks, tsd.PITCH, 25)
                rows.append((gt, blocks) + slim(pd_, ph, aux, pl, b))
                print("dev", v, g, seed, flush=True)
        out[v] = rows
    pickle.dump(out, open(path, "wb"))
    return out

def cache_dfl():
    path = os.path.join(HERE, "cache_dfl.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    import lightgbm as lgb
    folds = json.load(open(os.environ["LDE_FOLDS"]))
    meta = json.load(open(os.path.join(C, "learned_dfl_evidence/model_meta.json")))
    inv = {kind: d for _, (d, kind) in folds.items()}
    full = {k: lgb.Booster(model_file=os.path.join(C, "learned_dfl_evidence/model_%s.txt" % k)) for k in ("d", "ha", "c")}
    matches = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
    rows = []
    for gi, m in enumerate(matches):
        models = {"meta": meta}
        for k in ("d", "ha", "c"):
            models[k] = bench._FoldBooster(lgb.Booster(model_file=os.path.join(inv[k], "fold_%s_%d.txt" % (k, gi))), full[k].num_trees())
        for seed in bench.DFL_SEEDS:
            slots, st, ball, gt, blocks, sign = bench.dfl_prepared(m, seed)
            pd_, ph, aux = lp.frame_probs(slots, st, ball, sign, blocks, tsd.PITCH, 25, models=models)
            rows.append((gt, blocks) + slim(pd_, ph, aux, slots, ball))
            print("dfl", m, flush=True)
    pickle.dump(rows, open(path, "wb"))
    return rows

def score(rows, cfg):
    s, m, n = [], [], 0
    for gt, blocks, pd_, ph, aux in rows:
        pred = lp.decode(pd_, ph, aux, blocks, len(gt), 25, cfg=cfg)
        r = pm.evaluate(gt, pred, bootstrap_n=0)
        s.append(r["spell_f1_mean"]); m.append(r["macro_f1"]); n += r["n_pred_spells"]
    return np.mean(s), np.mean(m), n

if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    dev = cache_dev(); dfl = cache_dfl()
    for cfg in cfgs:
        out = []
        for v in ("clean", "offset", "osd", "stress2"):
            s, m, n = score(dev[v], cfg); out.append("%s %.4f/%.4f n%d" % (v, s, m, n))
        s, m, n = score(dfl, cfg); out.append("DFL %.4f/%.4f n%d" % (s, m, n))
        print(json.dumps(cfg), " | ".join(out), flush=True)
