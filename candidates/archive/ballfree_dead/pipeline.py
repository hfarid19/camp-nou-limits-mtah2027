"""ballfree_dead: a ball-free dead-ball / game-state model + LDE's team model, Viterbi-decoded.

See README.md for the method, references, data, variants and scores.

Dead evidence. model_bfd.txt is a LightGBM dead/alive classifier on features_bf.py: ~390
identity-free, team-swap- and pitch-mirror-invariant features computed from the PLAYER slots
only (the ball never enters): collective speeds / accelerations / velocity coherence, team
shape, set-piece configurations (kick-off split, centre circle, corner flag, touchline, box
crowding, a wall, tight marking) and tracked-player counts, with Lang et al. (2022)'s temporal
context (+-0.4 ... 80 s). Trained on the 7 open DFL/IDSSE games degraded with LDE's domain
randomisation (seeds 41-43), leave-one-game-out fold models for the DFL bench.

Team evidence + decoder. learned_dfl_evidence's model_ha / model_c / model_d (copied unchanged as
lde_model_*.txt, same sha256, fold-mapped through folds_verified.json) and its decoder
(lde_pipeline.py: contrast team logit, 3-state Viterbi, short-segment merge, hand-over snap).
The dead log-odds fed to the decoder are  w_bf * logit(p_bf) + w_lde * logit(p_lde_d)
(w_lde 0 = the purely ball-free dead model), plus LDE's ball-gap terms when enabled. Shipped:
w_bf = w_lde = 1, a_tcn = 0, ball_post = 1 (DFL LOGO spellF1 0.6283; pure ball-free V3 0.6088).
"""

import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "candidates"))     # dfl_folds
sys.path.insert(0, HERE)
import dfl_folds             # noqa: E402
import lde_pipeline as lde   # noqa: E402
import features_bf as fb     # noqa: E402
import tcn_np                # noqa: E402

warnings.filterwarnings("ignore")

CFG = dict(
    w_bf=1.0,          # weight of the ball-free dead logit
    w_lde=1.0,         # weight of LDE's (ball-using) dead logit (0: purely ball-free dead evidence)
    a_tcn=0.0,         # ball-free logit = (1 - a_tcn) * LightGBM + a_tcn * TCN (README: 0 here)
    bf_bias=0.0,       # added to the ball-free dead logit
    ball_post=1.0,     # 1: keep LDE's ball-based dead terms and snaps (w_gap, gap-end / out snaps)
)
_OV = sorted(k for k in CFG if "BFD_" + k.upper() in os.environ)
_LDE_OV = {}
if os.environ.get("BFD_SWEEP") == "1":       # tuning only
    for _k in _OV:
        CFG[_k] = float(os.environ["BFD_" + _k.upper()])
    for _k in lde.CFG:
        if "BFDL_" + _k.upper() in os.environ:
            _LDE_OV[_k] = float(os.environ["BFDL_" + _k.upper()])
    if _OV or _LDE_OV:
        print("[bfd] sweep overrides: {} {}".format({k: CFG[k] for k in _OV}, _LDE_OV),
              file=sys.stderr)

_M = {}
_CACHE = {}


def _bf_model():
    if "bf" not in _M:
        import hashlib
        import lightgbm as lgb
        path = os.path.join(HERE, "model_bfd.txt")
        with open(os.path.join(HERE, "model_bfd_meta.json")) as h:
            meta = json.load(h)
        with open(path, "rb") as h:
            assert hashlib.sha256(h.read()).hexdigest() == meta["sha256"], "model_bfd.txt changed"
        assert lgb.__version__ == meta["lgb_version"]
        _M["bf"] = lgb.Booster(model_file=path)
        _M["meta"] = meta
    return _M["bf"], _M["meta"]


def _tcn():
    if "tcn" not in _M:
        path = os.path.join(HERE, "tcn_bfd.npz")
        with open(os.path.join(HERE, "tcn_bfd_meta.json")) as h:
            meta = json.load(h)
        import hashlib
        with open(path, "rb") as h:
            assert hashlib.sha256(h.read()).hexdigest() == meta["sha256"], "tcn_bfd.npz changed"
        with np.load(dfl_folds.resolve(path)) as z:      # the LOGO fold file under bench --dfl
            _M["tcn"] = {k: z[k] for k in z.files}
    return _M["tcn"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ballfree_dead_logit(players, slot_team, pitch, fps):
    """Ball-free dead log-odds per frame: LightGBM on features_bf (+ the TCN if a_tcn > 0)."""
    model, meta = _bf_model()
    F, names = fb.compute(players, slot_team, pitch, fps)
    col = {n: i for i, n in enumerate(names)}
    l_gbm = _logit(model.predict(F[:, [col[n] for n in meta["names"]]], num_threads=1))
    if CFG["a_tcn"] <= 0:
        return l_gbm, l_gbm, None
    z = _tcn()
    n = len(F)
    t5 = np.arange(0, n, 5)
    X5 = F[t5][:, [col[str(nm)] for nm in z["names"]]]
    l_tcn = np.interp(np.arange(n), t5, tcn_np.logits(z, X5))
    return (1 - CFG["a_tcn"]) * l_gbm + CFG["a_tcn"] * l_tcn, l_gbm, l_tcn


def _decoder_cfg():
    c = dict(_LDE_OV)
    if CFG["ball_post"] < 0.5:      # purely ball-free dead evidence: no ball-gap / ball-out terms
        c.update(w_gap=0.0, w_out=0.0, post_v2=0.0, d_end_snap_s=0.0, d_start_lookback_s=0.0)
        for k, v in _LDE_OV.items():
            c[k] = v
    return c


def _run(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    key = (players.shape, players[::101].tobytes(), np.nan_to_num(ball[::101]).tobytes(),
           tuple(map(tuple, blocks)), len(tags))
    if key in _CACHE:
        return _CACHE[key]
    p_lde, p_home, aux = lde.frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    aux["long_gap"] = lde.long_gap_gated(players, ball, fps, lde.CFG["gap_s"], lde.CFG["gap_speed_ms"])
    aux["kick_restarts"] = lde.kickoff_restarts(tags, blocks, ball, fps, lde.CFG["prekick_s"])
    l_bf, l_gbm, l_tcn = ballfree_dead_logit(players, slot_team, pitch, fps)
    ld = CFG["w_bf"] * (l_bf + CFG["bf_bias"])
    if CFG["w_lde"] != 0:
        ld = ld + CFG["w_lde"] * _logit(p_lde)
    p_dead = 1 / (1 + np.exp(-ld))
    cfg = _decoder_cfg()
    codes = lde.decode(p_dead, p_home, aux, blocks, len(ball), fps, tags, names, cfg=cfg)
    ph, pdd = lde.evidence_probs(p_dead, p_home, aux, len(ball), fps, tags, names, cfg=cfg)
    proba = np.stack([ph * (1 - pdd), (1 - ph) * (1 - pdd), pdd], 1)
    _CACHE.clear()
    _CACHE[key] = (codes, proba, dict(l_bf=l_bf, l_gbm=l_gbm, l_tcn=l_tcn, p_lde=p_lde))
    return _CACHE[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)[1]
