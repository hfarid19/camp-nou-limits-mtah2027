"""Window-aggregated features for the team call, computed from LDE's per-frame features.

extra(F, names, fps) -> (E float32 (N, K), names, kinds), with kinds as in
learned_dfl_evidence/features.py ("sym", "anti", "H:<base>" / "A:<base>"), so the team-swap
augmentation and the symmetric inference cover them too.

A single tree cannot integrate evidence over time: every split looks at one frame's features.
LDE's features already carry centred moving means and shifted copies, but the only "who had
the ball" memory is the strict confident-control event (player within 1 m, nearest opponent
beyond 1.5 m, for 3 frames), which is rare under degraded tracking. These features add a looser
"near" event and backward / forward integrals of it:

- near_H / near_A: ball seen, that team's nearest player within NEAR_M and closer than the other
  team's by at least MARGIN_M.
- lastnear / nextnear: +1 home / -1 away / 0 none, the team of the last (next) near event.
- H_since_near / A_since_near (and until): log1p(frames) since that team's last near event.
- nearbal_b<w> / nearbal_f<w>: mean of (near_H - near_A) over the previous (next) w seconds.
- margin_b<w> / margin_f<w>: mean of clip(A_d1 - H_d1, -5, 5) m over frames where somebody is
  within 3 m of the ball, in the previous (next) w seconds (NaN if there is none).
"""

import numpy as np

NEAR_M, MARGIN_M, CLOSE_M = 2.0, 0.5, 3.0
WINDOWS = (1.0, 3.0, 8.0)


def _since(mask, cap):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.where(last >= 0, np.arange(n) - last, cap), cap).astype(np.float64)


def _last_code(code):
    n = len(code)
    idx = np.where(~np.isnan(code), np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, code[np.maximum(last, 0)], 0.0)


def _wmean(x, lo, hi):
    """NaN-aware mean of x[t+lo : t+hi+1]."""
    v = np.where(np.isnan(x), 0.0, x)
    c = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    n = len(x)
    i = np.arange(n)
    a, b = np.clip(i + lo, 0, n), np.clip(i + hi + 1, 0, n)
    cnt = cc[b] - cc[a]
    return np.where(cnt > 0, (cv[b] - cv[a]) / np.maximum(cnt, 1), np.nan)


def extra(F, names, fps=25):
    col = {k: i for i, k in enumerate(names)}
    seen = F[:, col["ball_seen"]] > 0.5
    dh = F[:, col["H_d1"]].astype(np.float64)
    da = F[:, col["A_d1"]].astype(np.float64)
    dh0, da0 = np.nan_to_num(dh, nan=99.0), np.nan_to_num(da, nan=99.0)
    nh = seen & (dh0 < NEAR_M) & (dh0 < da0 - MARGIN_M)
    na = seen & (da0 < NEAR_M) & (da0 < dh0 - MARGIN_M)
    cols, out_names, kinds = [], [], []

    def add(name, v, kind):
        cols.append(np.asarray(v, np.float32))
        out_names.append(name)
        kinds.append(kind)

    code = np.where(nh, 1.0, np.where(na, -1.0, np.nan))
    add("lastnear", _last_code(code), "anti")
    add("nextnear", _last_code(code[::-1])[::-1], "anti")
    cap = 30 * fps
    add("H_since_near", np.log1p(_since(nh, cap)), "H:since_near")
    add("A_since_near", np.log1p(_since(na, cap)), "A:since_near")
    add("H_until_near", np.log1p(_since(nh[::-1], cap)[::-1]), "H:until_near")
    add("A_until_near", np.log1p(_since(na[::-1], cap)[::-1]), "A:until_near")
    bal = nh.astype(np.float64) - na.astype(np.float64)
    close = np.fmin(dh, da) < CLOSE_M
    margin = np.where(close, np.clip(da - dh, -5, 5), np.nan)
    for w in WINDOWS:
        k = int(round(w * fps))
        add("nearbal_b{:g}".format(w), _wmean(bal, -k, 0), "anti")
        add("nearbal_f{:g}".format(w), _wmean(bal, 0, k), "anti")
        add("margin_b{:g}".format(w), _wmean(margin, -k, 0), "anti")
        add("margin_f{:g}".format(w), _wmean(margin, 0, k), "anti")
    E = np.stack(cols, 1).astype(np.float32)
    E[np.isinf(E)] = np.nan
    return E, out_names, kinds
