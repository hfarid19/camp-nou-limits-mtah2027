"""Cache LDE frame probabilities per bench row, clean and stressed (osmooth,drop / offset)."""
import os, sys, pickle, importlib.util
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench as B
import possession_pipeline as pp
import tune_on_sample_data as tsd
spec = importlib.util.spec_from_file_location("lde", os.path.join(ROOT, "candidates/learned_dfl_evidence/pipeline.py"))
L = importlib.util.module_from_spec(spec); sys.path.insert(0, os.path.join(ROOT, "candidates/learned_dfl_evidence")); spec.loader.exec_module(L)
OUT = os.environ.get("LDE_TUNE_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "tune"))
os.makedirs(OUT, exist_ok=True)

def _smooth(ball, hw=4):
    seen = ~np.isnan(ball[:, 0])
    o = np.stack([pp.moving_mean(ball[:, i].astype(np.float64), hw) for i in (0, 1)], 1)
    o[~seen] = np.nan
    return o.astype(np.float32)
def _offset(ball, rng, sd=1.3, tau=25):
    n = len(ball); a = np.exp(-1.0 / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), (n, 2)); off = np.zeros((n, 2)); off[0] = rng.normal(0, sd, 2)
    for t in range(1, n): off[t] = a * off[t - 1] + e[t]
    return (ball + off).astype(np.float32)
def _drop(players, rng, rate=0.06, mean_len=50):
    players = players.copy(); n, S, _ = players.shape
    starts = rng.random((n, S)) < rate / mean_len
    for s, j in zip(*np.where(starts)): players[s:s + int(rng.exponential(mean_len)) + 1, j] = np.nan
    return players
def stress(players, ball, mode):
    if mode == "offset": ball = _offset(ball.copy(), np.random.default_rng(7))
    if mode == "osd":
        ball = _smooth(_offset(ball.copy(), np.random.default_rng(7)), hw=6)
        players = _drop(players, np.random.default_rng(8))
    return players, ball

KEEP = ["H_d1", "A_d1", "ball_seen", "ball_out"]
MD = os.environ.get("MODEL_DIR")
MODELS = None
if MD:
    import lightgbm as lgb, json
    MODELS = dict(d=lgb.Booster(model_file=os.path.join(MD, "model_d.txt")),
                  ha=lgb.Booster(model_file=os.path.join(MD, "model_ha.txt")),
                  c=lgb.Booster(model_file=os.path.join(MD, "model_c.txt")),
                  meta=json.load(open(os.path.join(MD, "model_meta.json"))))
TAG = os.environ.get("TAG", "")
for mode in sys.argv[1:]:
    rows = []
    for name in B.DEV:
        for seed in B.SEEDS:
            slots, st, ball, gt, blocks, sign = B.prepared(name, seed)
            slots, ball = stress(slots, ball, mode)
            pdd, ph, aux = L.frame_probs(slots, st, ball, sign, blocks, tsd.PITCH, tsd.FPS, models=MODELS)
            col = [aux["names"].index(k) for k in KEEP]
            aux = dict(F=aux["F"][:, col], names=KEEP, p_change=aux["p_change"], change_prior=aux["change_prior"])
            rows.append(dict(name=name, seed=seed, gt=gt, blocks=blocks, pdd=pdd, ph=ph, aux=aux,
                             players=slots, ball=ball, st=st, sign=sign))
            print(mode, name, seed, flush=True)
    pickle.dump(rows, open(os.path.join(OUT, "rows_%s%s.pkl" % (mode, TAG)), "wb"))
