"""Per-frame signals and CRF feature groups for pathcrf_dynamic.

signals()  -> dict of named 1-D arrays: LDE's evidence (team / dead / hand-over log-odds) plus a
              few raw tracking columns of features.py. These are what prep.py stores, so the CRF
              features can be rebuilt without re-running LightGBM.
build()    -> dict of feature groups (float32, NaN-free) for the CRF potentials:

  ut    (N, .) antisymmetric, + = home      unary team score  T_t = w_ut . ut
  ud    (N, .) symmetric                    unary dead score  D_t = w_ud . ud
  ssym  (N, .) symmetric                    H<->A switch potential, shared part
  santi (N, .) antisymmetric, + = towards H switch potential, oriented to the receiving team
  die   (N, .) symmetric                    alive -> D potential
  rsym  (N, .) symmetric                    D -> alive potential, shared part
  ranti (N, .) antisymmetric, + = H         D -> alive potential, oriented to the restarting team
  lde   (N, 2) [lh, ld]                     LDE's evidence, for the fixed-unary CRF (crf.py)

A transition at frame t is the step y[t-1] -> y[t]; its features look both ways around t.
"""

import numpy as np

CLIP = 6.9


def _nan0(x, fill=0.0):
    x = np.asarray(x, np.float64)
    return np.where(np.isfinite(x), x, fill)


def shift(x, k):
    """y[t] = x[t + k], edge-padded."""
    n = len(x)
    idx = np.clip(np.arange(n) + k, 0, n - 1)
    return x[idx]


def mmean(x, hw):
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - hw), np.minimum(len(x), i + hw + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def mmax(x, hw):
    """Centred moving max (2*hw+1)."""
    n = len(x)
    out = x.copy()
    for k in range(1, hw + 1):
        out = np.maximum(out, shift(x, k))
        out = np.maximum(out, shift(x, -k))
    return out


def _logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


SIG_COLS = ["ball_seen", "ball_since", "ball_gap", "ball_speed", "ball_out", "diff_d1", "dmin",
            "mean_speed", "stationary", "lastctrl", "nextctrl"]


def signals(p_dead, p_home, p_change, change_prior, F, names, long_gap, lde_cfg):
    """LDE's evidence as log-odds + raw columns. lh_c is LDE's local-contrast team logit."""
    col = {k: i for i, k in enumerate(names)}
    q = np.clip(p_home, 1e-6, 1 - 1e-6)
    l6 = np.log(q / (1 - q))
    import possession_pipeline as pp
    cc = l6 - pp.moving_mean(l6, int(lde_cfg["contrast_hw"]))
    cc = np.sign(cc) * np.maximum(np.abs(cc) - lde_cfg["contrast_gate"], 0.0)
    lh_c = np.clip(l6 + lde_cfg["contrast_beta"] * cc, -CLIP, CLIP)
    sig = dict(
        lr=np.clip(_logit(p_home), -CLIP, CLIP),
        lh=lh_c,
        ld=np.clip(_logit(p_dead) + lde_cfg["w_gap"] * long_gap, -CLIP, CLIP),
        lc=np.clip(_logit(p_change) - _logit(np.array([change_prior]))[0], -CLIP, CLIP),
        long_gap=np.asarray(long_gap, np.float64),
    )
    for k in SIG_COLS:
        sig[k] = F[:, col[k]].astype(np.float64)
    return {k: np.asarray(v, np.float32) for k, v in sig.items()}


def build(sig):
    g = {k: np.asarray(v, np.float64) for k, v in sig.items()}
    lr, lh, ld, lc = g["lr"], g["lh"], g["ld"], g["lc"]
    seen = _nan0(g["ball_seen"])
    spd = np.clip(_nan0(g["ball_speed"]), 0, 40) / 10.0
    gap = _nan0(g["ball_gap"]) / 4.0
    since = _nan0(g["ball_since"]) / 4.0
    out = (_nan0(g["ball_out"]) > 0).astype(np.float64)
    out_s = mmean(out, 2)
    dd1 = np.clip(_nan0(g["diff_d1"]), -20, 20) / 10.0          # H_d1 - A_d1 (+ = away closer)
    dmin = np.clip(_nan0(g["dmin"], 20.0), 0, 20) / 10.0
    msp = np.clip(_nan0(g["mean_speed"], 0.0), 0, 8) / 4.0
    stat = _nan0(g["stationary"], 0.5)
    lctl, nctl = _nan0(g["lastctrl"]), _nan0(g["nextctrl"])
    lg = g["long_gap"]

    def dlt(x, k):
        return shift(x, k) - shift(x, -k)
    feats = dict(
        ut=np.stack([lr, lh, lr * seen, mmean(lr, 12), lctl, nctl], 1),
        ud=np.stack([np.ones_like(ld), ld, mmean(ld, 12), ld * seen, lg, out_s], 1),
        ssym=np.stack([np.ones_like(lc), lc, lc - mmax(lc, 12), lc * seen, seen, spd,
                       np.abs(dlt(spd, 3)), gap, dmin, np.abs(dlt(lh, 6)) / 4.0, ld / 4.0], 1),
        santi=np.stack([dlt(lh, 6) / 4.0, dlt(lh, 25) / 4.0, -dd1, (nctl - lctl) / 2.0,
                        lr / 4.0], 1),
        die=np.stack([np.ones_like(ld), ld / 4.0, dlt(ld, 6) / 4.0, dlt(ld, 25) / 4.0, out_s,
                      out_s - shift(out_s, -12), seen, gap, msp, stat, lg], 1),
        rsym=np.stack([np.ones_like(ld), ld / 4.0, -dlt(ld, 6) / 4.0, -dlt(ld, 25) / 4.0, seen,
                       since, msp, stat, lg, out_s], 1),
        ranti=np.stack([lh / 4.0, shift(lh, 12) / 4.0, lr / 4.0], 1),
        lde=np.stack([lh, ld], 1),                 # LDE's team / dead log-odds (fixed unary)
    )
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for k, v in feats.items()}


GROUPS = ["ut", "ud", "ssym", "santi", "die", "rsym", "ranti", "lde"]
