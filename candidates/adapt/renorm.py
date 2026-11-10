"""Label-free per-game feature re-normalisation (round 5, track R5-2).

The NN members z-score their 194 features with the training mean / sd stored in their weight file.
Here the training statistics are moved towards the statistics of the game being labelled:

    mu' = mu_tr + alpha * m * (mu_g - mu_tr)
    sd' = sd_tr * exp(alpha * m * log(clip(sd_g / sd_tr, 1 / RMAX, RMAX)))

mu_g, sd_g: the game's own feature moments over its graded block frames, pooled with the team-swapped
copy of the features (fe.swap_index), so the transform is mirror-equivariant: the members' mirror pass
gets exactly the same normalisation (anti-symmetric features keep a zero mean, H/A pairs share stats).
m: per-feature mask (all features, a fixed list, or the features whose game moments are far from the
training moments). No labels; deterministic numpy only.
"""

import numpy as np

RMAX = 3.0
MIN_N = 500          # a feature needs this many non-NaN graded frames to be re-normalised


def block_rows(n, blocks):
    m = np.zeros(n, bool)
    for _, first, last in blocks:
        m[int(first):int(last) + 1] = True
    return m


def game_moments(F, blocks, perm, signs):
    """-> (mean, sd, count) per feature, mirror-pooled, over graded block frames (NaN ignored)."""
    X = np.asarray(F[block_rows(len(F), blocks)], np.float64)
    ok = ~np.isnan(X)
    Xz = np.where(ok, X, 0.0)
    n = ok.sum(0).astype(np.float64)
    s1 = Xz.sum(0)
    s2 = (Xz * Xz).sum(0)
    sg = np.asarray(signs, np.float64)
    n_p = n + n[perm]
    s1_p = s1 + s1[perm] * sg
    s2_p = s2 + s2[perm]
    mean = s1_p / np.maximum(n_p, 1.0)
    var = np.maximum(s2_p / np.maximum(n_p, 1.0) - mean * mean, 0.0)
    return mean, np.sqrt(var), n_p / 2.0


def adapted(mu_tr, sd_tr, mom, alpha, mask=None, auto_thr=None, mean_only=False, perm=None):
    """(mu', sd') for one model. mask: bool (K,) or None (all). auto_thr = (smd, log_ratio): restrict
    to features whose |mu_g - mu_tr| / sd_tr > smd or |log(sd_g / sd_tr)| > log_ratio."""
    mu_tr = np.asarray(mu_tr, np.float64).reshape(-1)
    sd_tr = np.asarray(sd_tr, np.float64).reshape(-1)
    mean, sd, cnt = mom
    m = np.ones(len(mu_tr), bool) if mask is None else np.asarray(mask, bool).copy()
    m &= cnt >= MIN_N
    m &= sd > 1e-6 * np.maximum(sd_tr, 1e-12)
    ratio = np.clip(sd / sd_tr, 1.0 / RMAX, RMAX)
    if auto_thr is not None:
        smd = np.abs(mean - mu_tr) / sd_tr
        m &= (smd > auto_thr[0]) | (np.abs(np.log(ratio)) > auto_thr[1])
    if perm is not None:                       # mirror closure: an H_ feature and its A_ partner
        m = m | m[np.asarray(perm)]
        m &= (cnt >= MIN_N) & (sd > 1e-6 * np.maximum(sd_tr, 1e-12))
    w = alpha * m.astype(np.float64)
    mu = mu_tr + w * (mean - mu_tr)
    sdn = sd_tr if mean_only else sd_tr * np.exp(w * np.log(ratio))
    return mu, sdn, m


# variants: name -> kwargs of adapted() (mask "flag" = FLAGGED); fixed before any result (see
# $MTAH_WORK/arch/r5_adapt/prereg_adapt.txt). "base" must reproduce the members' own codes.
VARIANTS = {
    "base": dict(alpha=0.0),
    "all_a25": dict(alpha=0.25),
    "all_a50": dict(alpha=0.5),
    "all_a100": dict(alpha=1.0),
    "mean_a100": dict(alpha=1.0, mean_only=True),
    "flag_a50": dict(alpha=0.5, mask="flag"),
    "flag_a100": dict(alpha=1.0, mask="flag"),
    "auto_a100": dict(alpha=1.0, auto_thr=(0.25, np.log(1.25))),
}

# features shifted beyond the DFL game-to-game range on the competition match (KS ratio > 1,
# $MTAH_WORK/stats/shift_features.csv, candidates/stats_report s06_shift.py): names only.
FLAGGED = [
    'ball_speed', 'ball_disp_b', 'ball_disp_f', 'n_tracked', 'stationary', 'box_homedef', 'H_length',
    'A_length', 'H_fwd', 'A_fwd', 'H_ntr', 'diff_spread', 'diff_length', 'diff_fwd', 'diff_close',
    'ball_seen_pmf', 'ball_speed_m0.5', 'ball_speed_m2.0', 'ball_speed_m5.0', 'ball_speed_pmf',
    'ball_out_pmf', 'stationary_m0.5', 'stationary_m2.0', 'stationary_m5.0', 'n_tracked_m0.5',
    'n_tracked_m2.0', 'n_tracked_m5.0', 'n_tracked_pmf', 'diff_c5_pmf', 'diff_fwd_m0.5',
    'diff_fwd_m2.0', 'diff_fwd_m5.0', 'diff_spread_m0.5', 'diff_spread_m2.0', 'diff_spread_m5.0',
    'ball_xdir_pmf', 'stationary_s+0.8', 'stationary_s-0.8', 'stationary_s+2.0', 'stationary_s-2.0',
    'stationary_s+4.0', 'stationary_s-4.0', 'stationary_s+20.0', 'stationary_s-20.0',
    'n_tracked_s+0.8', 'n_tracked_s-0.8', 'n_tracked_s+2.0', 'n_tracked_s-2.0', 'n_tracked_s+4.0',
    'n_tracked_s-4.0', 'n_tracked_s+20.0', 'n_tracked_s-20.0']


def variant_stats(name, mu_tr, sd_tr, mom, names, perm):
    kw = dict(VARIANTS[name])
    if kw.get("mask") == "flag":
        fl = set(FLAGGED)
        kw["mask"] = np.array([str(s) in fl for s in names])
    return adapted(mu_tr, sd_tr, mom, perm=perm, **kw)
