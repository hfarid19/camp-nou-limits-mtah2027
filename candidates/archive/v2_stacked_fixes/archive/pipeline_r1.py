"""v2_stacked_fixes: v1 pipeline plus the verified review fixes, stacked one step at a time.

ROUND 1 (after the 5 judges) -- shipped defaults now differ from the R0 log below:
  S9 team evidence = the learned DFL H/A logit ONLY (frozen snapshot in lde_frozen/ of
     learned_dfl_evidence's features.py + model_ha.txt + model_d.txt, trained only on the open
     DFL/IDSSE tracking of other games; needs lightgbm 4.7.0). team_logit = w_lde*clip(logit
     p_home, +-5) + tag priors, w_lde 6. v1's ball/shape terms and S1's carry are off
     (lastctrl_w 0, w_shape = w_ball_near = w_ball_far = 0). Dead model (S2) unchanged.
  Proximity post-rules off: pp.snap_boundaries (snap_v1 0) and prune_uncontrolled (prune 0).
  k deadband removed (k_deadband 0). Rules: V2_CFG only with V2_SWEEP=1 (warns), team names
  matched only against the meta names, prekick tolerance generic (prekick_tol_s), prekick only
  on 'kick off (start)' tags.
  Selection on clean bench, checked on offset (AR(1) sd 1.3 m) and stress (smooth+offset+drop,
  candidates/_scratch/judge_generalisation-risk/stress.py) -- see README.md for the table.
  R1 dev: spellF1 0.6468, macroF1 0.8152 (R0 0.6048 / 0.8086); offset 0.6050, stress 0.5855
  (R0 0.4737, 0.4549). Match: 303 possessions, 45.5% dead, md5 872feef4723b8ed1975091b4727e88b6.
  R0 is preserved as pipeline_r0.py.

R0 docstring follows.

Fork of possession_pipeline.build_evidence (local build_evidence below); pp.viterbi,
pp.fit_logistic, pp.moving_mean, pp.runs_of, pp.merge_short, pp.snap_boundaries reused unchanged.
Every new knob is in CFG (env V2_CFG json overrides it, for sweeps only; the defaults are the
chosen values). numpy only, deterministic (no RNG).

Steps (all on by default unless noted)
  S1 carry-forward control prior: last confident control (raw ball seen, own nearest < ctrl_r*k,
     other team > ctrl_margin*k, held ctrl_persist=3 consecutive frames) forward-filled per block,
     +-lastctrl_w=2.0 added to the team logit.
  S2 dead model: -1.5 + 4 out + 2 (2 slow-1) + 4 few + 4 long_gap_gated (raw ball gap > 3 s with
     mean player speed < 2 m/s) + 0.8 static_disp (|b(t+12)-b(t-12)| < 0.5 m) + 1.0 tag_dead.
  S3 snaps: pp.snap_boundaries unchanged, then each D end -> end of a long ball gap within 4 s,
     then each alive->D change -> first ball-out frame in the previous 12 s (no fallback).
     The gap snap of D *starts* is implemented but OFF (snap_gap_start=0): it lowered spellF1.
  S4 merge_short_v2 -> prune_uncontrolled -> merge_short_v2 (prune still helps on top of S1).
  S5 radii x k, k = clip(q61(dmin | raw ball seen, graded, mean speed > 1.4)/1.5, 1, 2), with a
     deadband: k < 1.1 -> 1.0 (q61 estimation noise; on the bench raw k was 1.000-1.008, and the
     1.008 row moved by -0.007 for no reason). Match k = 1.722.
  S6 5-frame NaN-aware moving mean of the raw ball, used only for displacement features.
  S7 Smart Tagging (match only; asserted to be a no-op without tags, bench predictions verified
     bit-identical with tag_fix 0/1): set-piece restart detection, tag_dead [s-1 s, restart),
     taking-team prior 10 s after the restart (nearest player within 2k m for n/a teams), goal
     D on the long ball gap ending nearest the tag end, pre-kick-off D (a 'kick off (start)'
     tag starting within 180 s of a block start; Smart Tagging timestamps are approximate, so
     a generic tolerance of prekick_tol_s=10 s before the block start is allowed), phase priors only from
     build up / progression / defending in attacking or middle third, sign of the net vote,
     w_tag=0.2; set-piece prior w_tag_sp=0.8 (v1's effective weight).
  S8 shape_lead: tested, +0.0035 < the required +0.005 -> OFF (shape_lead=0).

Build log (bench dev = Sample Games 1, 2 x seeds 2027, 2028; rows in that order)
  step                         spellF1  macroF1  g1     g2     rows spellF1                  pred poss (gt 348,348,329,329)  kept
  v1 (fork, all off)           0.4749   0.7453   .4930  .4568  .5002 .4858 .4591 .4546   279 277 253 250   reference
  S1 w2.0 r0.75 (default)      0.5557   0.7630   .5554  .5559  .5474 .5635 .5483 .5635   287 293 251 263   yes
     sweep w1.5 r0.75          0.5508   0.7612   .5571  .5445                             286 296 255 258
     sweep w2.5 r0.75          0.5568   0.7628   .5563  .5573                             295 301 256 268
     sweep w1.5 r1.0           0.5424   0.7547                                           305 316 281 286
     sweep w2.0 r1.0           0.5524   0.7568                                           307 313 279 289
     sweep w2.5 r1.0           0.5571   0.7566                                           318 334 281 293
     -> plateau w 2.0-2.5 at r 0.75; middle = w 2.0, r 0.75
  S2 wl4 ws2 (default)         0.5948   0.8073   .5805  .6091  .5689 .5922 .6042 .6140   282 289 245 256   yes
     sweep wl3 ws2 / ws2.5     0.5946 / 0.5880   macro 0.8058 / 0.7980
     sweep wl4 ws2.5           0.5938   0.8020
     sweep wl5 ws2 / ws2.5     0.5948 / 0.5938   (identical to wl4)
  S3 gap snap start+end        0.5946   0.8092   (spellF1 -0.0002)
     gap snap start only       0.5943   0.8074   -> dropped
     out look-back snap only   0.5948   0.8073   (no-op on bench: ball out on 1.3% / 0.2% of frames)
  S3 gap end + out look-back   0.5952   0.8091   .5809  .6095  .5702 .5915 .6042 .6147   282 289 245 256   yes
  S4 merge_v2 only             0.6003   0.8085   .5842  .6164                            265 274 237 245
  S4 merge_v2+prune+merge_v2   0.6046   0.8083   .5867  .6226  .5770 .5964 .6157 .6295   262 271 233 241   yes
  S5 k without deadband        0.6029   0.8083   .5832  .6226  .5770 .5893 .6157 .6295   262 269 233 241   (k 1.000/1.008/1.000/1.000)
  S5 k with deadband 0.1       0.6046   0.8083   identical to S4                         262 271 233 241   yes
     AR(1) offset check (robust_offset.py, sd 1.3 m, tau 25, seed 7): S4 no k 0.4314 / 0.7634;
     S5 with k 0.4737 / 0.7756 (k 1.64) -> k helps under real-like offsets
  S6 ball smoothing 5 frames   0.6048   0.8086   .5867  .6229  .5770 .5964 .6164 .6295   262 271 233 241   yes
  S7 tag fixes                 0.6048   0.8086   bit-identical on the bench              262 271 233 241   yes
  S8 shape_lead 25 / 12        0.6083 / 0.6078   macro 0.8093 / 0.8089  (+0.0035 < 0.005)               no
  Extra probes on the final stack (not adopted): switch_cost 1.5/1.75/2.0/2.5/2.75 ->
     0.6065/0.6040/0.6048/0.6044/0.6011 (flat); dead costs 4/6 -> 0.6049/0.6045; min_poss_s
     0.6/1.4 -> 0.5954/0.6046; temperature 0.12/0.18 -> 0.6005/0.6041; ctrl_persist 1/2/5 ->
     0.6080/0.6112/0.5715 on the bench, but under the AR(1) offset kp 1/2/3 -> 0.4563/0.4667/
     0.4737, so ctrl_persist stays 3 (more robust to the match's larger ball offsets).

Match (--match, no score): before S7 325 possessions, dead 45.0%; after S7 (final) 322
possessions, mean 10.7 s, dead 45.7%, k 1.722. S7 changes 4015 frames, mostly the 2nd-half
pre-kick-off window (was 1797 H / 116 A / 1104 D frames, now all D). submission.csv is
byte-identical across two runs (md5 35c389233f90899963ea4abe695bd8b2).

FINAL dev RESULT: spellF1_mean 0.6048, macroF1_mean 0.8086; rows 0.577 / 0.5964 / 0.6164 /
0.6295 (all above v1's 0.5002 / 0.4858 / 0.4591 / 0.4546).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import possession_pipeline as pp  # noqa: E402
from possession_pipeline import (fill_short_gaps, fit_logistic, moving_mean, runs_of,  # noqa: E402
                                 segments, sigmoid)

H, A, D = 0, 1, 2

CFG = dict(
    # S1 carry-forward control prior
    lastctrl_w=0.0, ctrl_r=0.75,   # R1: 0 (was 2.0), the learned team evidence replaces it
    ctrl_margin=1.5, ctrl_persist=3,
    # S2 dead model
    dead_v2=1, w_long=4.0, w_slow=2.0, long_s=3.0, lm_gate=2.0, w_static=0.8,
    static_half_s=0.48, static_disp_m=0.5,
    # S3 boundary snaps
    snap_gap=1, snap_gap_start=0, snap_gap_end=1, snap_win_s=4.0, out_lookback_s=12.0, snap_back=1, keep_s=0.5,
    # S4 segment hygiene
    hygiene=1, prune=0, prune_rounds=3, v1_merge=1, snap_v1=0,   # R1: prune, snap_v1 off
    # S5 match-adaptive radii
    adapt_k=1, k_q=61, k_ref_m=1.5, k_lo=1.0, k_hi=2.0, k_deadband=0.0, k_scale_all=0,
    # S6 ball smoothing for kinematic features (half width in frames: 2 -> 5-frame window)
    ball_smooth_hw=2,
    # S7 Smart Tagging fixes (no-op without tags)
    tag_fix=1, sp_horizon_s=20.0, sp_missing_s=1.0, sp_move_m=1.0, sp_move_s=0.5,
    sp_fallback_s=5.0, sp_prior_s=10.0, w_tag_sp=0.8, goal_win_s=15.0, prekick_s=180.0,
    prekick_tol_s=10.0, w_tag=0.2,
    # S8 shape lead (frames)
    shape_lead=0,
    # S9 learned DFL team evidence (learned_dfl_evidence model_ha.txt), added to the team logit
    w_lde=6.0, lde_clip=5.0, w_lde_dead=0.0,
    # overrides of possession_pipeline.PARAMS (decoder costs etc.). R1: v1's in-match ball and
    # shape team terms off; the team evidence is the learned DFL logit alone (S9).
    pp_over=dict(w_shape=0.0, w_ball_near=0.0, w_ball_far=0.0),
)
if os.environ.get("V2_SWEEP") == "1" and os.environ.get("V2_CFG"):   # sweeps only (opt-in)
    CFG.update(json.loads(os.environ["V2_CFG"]))
    print("[v2] WARNING: V2_CFG override active (V2_SWEEP=1): " + os.environ["V2_CFG"],
          file=sys.stderr)
elif os.environ.get("V2_CFG"):
    print("[v2] WARNING: V2_CFG is set but ignored (set V2_SWEEP=1 to apply it)", file=sys.stderr)

PHASE_POSS = {"build up", "progression"}
PHASE_DEF = {"defending in attacking third", "defending in middle third"}


def _log(*a):
    if os.environ.get("V2_QUIET") != "1":
        print("[v2]", *a, file=sys.stderr)


def _shift_fwd(F, k):
    if k <= 0:
        return F
    out = np.empty_like(F)
    out[:-k] = F[k:]
    out[-k:] = F[-1:]
    return out


def _side(team, names):
    """+1 home, -1 away, 0 unknown. Matched only against the meta team names: exact
    (case-insensitive), else a generic fuzzy match (difflib ratio > 0.8, unambiguous)."""
    import difflib
    home_name, away_name = names[0].strip().lower(), names[1].strip().lower()
    team = (team or "").strip().lower()
    if not team or team in ("n/a", "na"):
        return 0.0
    if team == home_name:
        return 1.0
    if team == away_name:
        return -1.0
    rh = difflib.SequenceMatcher(None, team, home_name).ratio()
    ra = difflib.SequenceMatcher(None, team, away_name).ratio()
    if max(rh, ra) > 0.8 and abs(rh - ra) > 0.1:
        return 1.0 if rh > ra else -1.0
    return 0.0


def _disp(b, k):
    """|b(t+k) - b(t-k)| with NaN where either end is missing."""
    n = len(b)
    out = np.full(n, np.nan)
    if n > 2 * k:
        out[k:n - k] = np.sqrt(((b[2 * k:] - b[:n - 2 * k]) ** 2).sum(1))
    return out


def _restart(raw_seen, bsm, s, horizon, miss_k, move_k, move_m, fallback, n):
    """First frame in [s, s+horizon) where the raw ball reappears after >= miss_k missing frames,
    or a seen stationary ball starts moving (> move_m over move_k frames). Fallback s+fallback."""
    for t in range(max(s, 1), min(n - move_k, s + horizon)):
        if raw_seen[t]:
            if t >= miss_k and not raw_seen[t - miss_k:t].any():
                return t
            if t >= move_k and raw_seen[t - move_k] and raw_seen[t + move_k]:
                before = np.hypot(*(bsm[t] - bsm[t - move_k]))
                after = np.hypot(*(bsm[t + move_k] - bsm[t]))
                if before < 0.5 and after > move_m:
                    return t
    return min(n, s + fallback)


_LDE = {}
LDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lde_frozen")


def _lde_logits(players, slot_team, ball, sign_half1, blocks, pitch, fps, clip):
    """Clipped log-odds of the learned DFL frame models: a frozen snapshot (lde_frozen/, md5s in
    README) of learned_dfl_evidence's features.py, model_d.txt and model_ha.txt, trained only on
    the open DFL/IDSSE tracking of OTHER games (CC BY 4.0, fetched by
    candidates/learned_dfl_evidence/fetch_idsse.py). Same inference as that candidate's
    frame_probs: H/A averaged with the team-swapped features, so it is exactly symmetric."""
    if "fe" not in _LDE:
        import importlib.util
        import lightgbm as lgb
        spec = importlib.util.spec_from_file_location("v2_lde_features",
                                                      os.path.join(LDE_DIR, "features.py"))
        fe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fe)
        with open(os.path.join(LDE_DIR, "model_meta.json")) as handle:
            meta = json.load(handle)
        if lgb.__version__ != meta.get("lgb_version", lgb.__version__):
            print("[v2] WARNING: lightgbm {} installed, models trained with {}".format(
                lgb.__version__, meta["lgb_version"]), file=sys.stderr)
        _LDE.update(fe=fe, meta=meta,
                    d=lgb.Booster(model_file=os.path.join(LDE_DIR, "model_d.txt")),
                    ha=lgb.Booster(model_file=os.path.join(LDE_DIR, "model_ha.txt")))
    fe = _LDE["fe"]
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, fnames, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    assert fnames == _LDE["meta"]["names"], "feature list differs from the frozen model"
    p_dead = _LDE["d"].predict(F, num_threads=1)
    perm, signs = fe.swap_index(fnames, kinds)
    p_home = 0.5 * (_LDE["ha"].predict(F, num_threads=1)
                    + 1.0 - _LDE["ha"].predict(F[:, perm] * signs, num_threads=1))
    lg = lambda p: np.clip(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)),
                           -clip, clip)
    return lg(np.asarray(p_dead, float)), lg(np.asarray(p_home, float))


def build_evidence(players, ball_raw, slot_team, sign_half1, blocks, tags, names, pitch, P, C):
    FPS = pp.FPS
    length, width = pitch
    n = len(ball_raw)
    home, away = slot_team == 0, slot_team == 1

    sign = np.full(n, sign_half1)
    for half, first, last in blocks:
        if half == 2:
            sign[first:] = -sign_half1
    block_id = np.full(n, -1)
    graded = np.zeros(n, bool)
    for i, (_, first, last) in enumerate(blocks):
        block_id[first:last + 1] = i
        graded[first:last + 1] = True

    gap = int(round(P["ball_gap_fill_s"] * FPS))
    ball = np.stack([fill_short_gaps(ball_raw[:, 0].astype(np.float64), gap),
                     fill_short_gaps(ball_raw[:, 1].astype(np.float64), gap)], 1)
    raw_seen = ~np.isnan(ball_raw[:, 0])
    seen = ~np.isnan(ball[:, 0])

    # S6: smoothed raw ball, for kinematic features only (speed / displacement)
    hw = int(C["ball_smooth_hw"])
    braw = ball_raw.astype(np.float64)
    if hw > 0:
        bsm = np.stack([moving_mean(braw[:, 0], hw), moving_mean(braw[:, 1], hw)], 1)
        bsm[~raw_seen] = np.nan
    else:
        bsm = braw

    dist = np.sqrt(((players - ball[:, None, :].astype(np.float32)) ** 2).sum(-1))
    d_home = np.fmin.reduce(dist[:, home], axis=1)
    d_away = np.fmin.reduce(dist[:, away], axis=1)
    del dist

    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * FPS
    spd = np.sqrt((vel ** 2).sum(-1))
    spd[spd > 12.0] = np.nan
    mean_speed = moving_mean(np.nanmean(spd, axis=1), FPS)
    n_tracked = moving_mean((~np.isnan(players[:, :, 0])).sum(1).astype(np.float64), FPS)

    # S5: label-free scale factor for the radii
    dmin = np.fmin(d_home, d_away)
    k = 1.0
    if C["adapt_k"]:
        sel = raw_seen & graded & (np.nan_to_num(mean_speed, nan=0.0) > P["dead_speed_ms"]) \
            & ~np.isnan(dmin)
        if sel.sum() > 100:
            k = float(np.clip(np.percentile(dmin[sel], C["k_q"]) / C["k_ref_m"],
                              C["k_lo"], C["k_hi"]))
            if k < 1.0 + C["k_deadband"]:        # within estimation noise of 1: no rescale
                k = 1.0
    _log("k = {:.3f}".format(k))
    P = dict(P)
    P["control_radius_m"] *= k
    P["near_ball_m"] *= k
    if C["k_scale_all"]:
        P["ball_scale_m"] *= k
        P["far_ball_m"] *= k
    ctrl_r, ctrl_margin = C["ctrl_r"] * k, C["ctrl_margin"] * k

    # Team shape
    x_dir = (players[:, :, 0] - length / 2.0) * sign[:, None]

    def team_stats(mask, s):
        xs, ys = x_dir[:, mask] * s, players[:, mask, 1]
        vx = vel[:, mask, 0] * sign[:, None] * s
        return (np.nanmean(xs, 1), np.nanstd(xs, 1) + np.nanstd(ys, 1),
                np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1),
                np.nanmean(np.clip(vx, -10, 10), 1))
    hc, hs, hl, hw_, hv = team_stats(home, 1.0)
    ac, as_, al, aw, av = team_stats(away, -1.0)
    raw_feats = [np.nanmean(x_dir, 1), hc + ac, hs - as_, hl - al, hw_ - aw, hv - av]
    F = np.stack([moving_mean(f, FPS) for f in raw_feats], 1)
    F = _shift_fwd(F, int(C["shape_lead"]))
    F_ok = ~np.isnan(F).any(1)

    r = P["control_radius_m"]
    alive_like = mean_speed > P["dead_speed_ms"]
    ph_home = raw_seen & (d_home < r) & (d_away > 2 * r) & alive_like & F_ok
    ph_away = raw_seen & (d_away < r) & (d_home > 2 * r) & alive_like & F_ok
    idx = np.where(ph_home | ph_away)[0]
    p_shape = np.full(n, 0.5)
    if len(idx) > 500 and ph_home.sum() > 100 and ph_away.sum() > 100:
        mu, sd = F[idx].mean(0), F[idx].std(0) + 1e-6
        Z = (F[idx] - mu) / sd
        w, b = fit_logistic(Z, ph_home[idx].astype(float))
        Zall = (np.where(F_ok[:, None], F, mu) - mu) / sd
        p_shape = np.where(F_ok, sigmoid(Zall @ w + b), 0.5)

    # Dead-ball cues
    margin = 0.5
    out = seen & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin) |
                  (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    out = moving_mean(out.astype(float), 3) > 0.5
    slow = sigmoid((P["dead_speed_ms"] - np.nan_to_num(mean_speed, nan=0.0)) / 0.3)
    few = np.nan_to_num(n_tracked, nan=0.0) < P["min_tracked"]

    gaps = []                                  # long raw-ball gaps (start, end)
    long_gated = np.zeros(n, bool)
    long_any = np.zeros(n, bool)
    for s, e in zip(*runs_of(~raw_seen)):
        if e - s > C["long_s"] * FPS:
            gaps.append((int(s), int(e)))
            long_any[s:e] = True
            if np.nanmean(mean_speed[s:e]) < C["lm_gate"]:
                long_gated[s:e] = True

    ks = int(round(C["static_half_s"] * FPS))
    static_disp = np.nan_to_num(_disp(bsm, ks), nan=99.0) < C["static_disp_m"]

    # Smart Tagging
    tag_team = np.zeros(n)
    tag_dead = np.zeros(n)
    kick_restarts = []                          # (tag start, restart) of kick-off set pieces
    if not C["tag_fix"]:
        v1_poss, v1_def = pp.POSSESSING_CODES, pp.DEFENDING_CODES
        for code, s, e, team, kind in tags:
            s, e = max(0, s), min(n, max(e, s + 1))
            t = _side(team, names)
            if code in v1_poss:
                tag_team[s:e] += t
            elif code in v1_def:
                tag_team[s:e] -= t
            elif code == "set pieces":
                cut = s + int(0.4 * (e - s))
                tag_dead[s:cut] += 1.0
                tag_team[cut:e] += t
            elif code == "goals":
                tag_dead[e:min(n, e + 10 * FPS)] += 1.0
        tag_team = np.clip(tag_team, -1, 1) * 2.0 * P["w_tag"]
        tag_dead = np.clip(tag_dead, 0, 1)
    else:
        phase = np.zeros(n)
        sp_team = np.zeros(n)
        block_first = {i: f for i, (_, f, _) in enumerate(blocks)}
        for code, s, e, team, kind in tags:
            s, e = max(0, s), min(n, max(e, s + 1))
            t = _side(team, names)
            if code in PHASE_POSS:
                phase[s:e] += t
            elif code in PHASE_DEF:
                phase[s:e] -= t
            elif code == "set pieces":
                rs = _restart(raw_seen, bsm, s, int(C["sp_horizon_s"] * FPS),
                              int(C["sp_missing_s"] * FPS), int(C["sp_move_s"] * FPS),
                              C["sp_move_m"], int(C["sp_fallback_s"] * FPS), n)
                bi = block_id[min(max(s, 0), n - 1)]
                if bi < 0:                       # tag starts just before a block: use next block
                    later = [f for f in block_first.values() if f >= s]
                    bstart = min(later) if later else 0
                else:
                    bstart = block_first[bi]
                tag_dead[max(bstart, s - FPS):rs] += 1.0
                if t == 0.0 and rs < n and raw_seen[rs]:
                    if np.nan_to_num(min(d_home[rs], d_away[rs]), nan=99.0) < 2.0 * k:
                        t = 1.0 if d_home[rs] <= d_away[rs] else -1.0
                sp_team[rs:min(e, rs + int(C["sp_prior_s"] * FPS))] += t
                if kind.strip().lower().startswith("kick off (start)"):
                    kick_restarts.append((s, rs))
            elif code == "goals":
                ends = [g for g in gaps if abs(g[1] - e) < C["goal_win_s"] * FPS]
                if ends:
                    g0 = min(ends, key=lambda g: abs(g[1] - e))
                    tag_dead[g0[0]:g0[1]] += 1.0
        # each phase counted once: sign of the net vote, weight w_tag; set-piece prior separate
        tag_team = C["w_tag"] * np.sign(phase) + C["w_tag_sp"] * np.clip(sp_team, -1, 1)
        tag_dead = np.clip(tag_dead, 0, 1)
    if not tags:
        assert not tag_team.any() and not tag_dead.any() and not kick_restarts

    if C["dead_v2"]:
        dead_logit = (-1.5 + 4.0 * out + C["w_slow"] * (2 * slow - 1) + 4.0 * few
                      + C["w_long"] * long_gated + C["w_static"] * static_disp
                      + P["w_tag_dead"] * tag_dead)
    else:
        bvel = np.full(n, np.nan)
        bvel[1:] = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(1)) * FPS
        ball_speed = moving_mean(bvel, FPS // 2)
        static = seen & (np.nan_to_num(ball_speed, nan=9.0) < 0.3)
        long_missing = np.zeros(n, bool)
        for s, e in zip(*runs_of(~seen)):
            if e - s > 3 * FPS:
                long_missing[s:e] = True
        dead_logit = (-1.5 + 4.0 * out + 3.5 * (2 * slow - 1) + 4.0 * few + 1.0 * long_missing
                      + 0.8 * static + P["w_tag_dead"] * tag_dead)
    p_dead = sigmoid(dead_logit)

    # Team evidence
    have_ball = seen & ~np.isnan(dmin)
    l_ball = np.clip(np.nan_to_num((d_away - d_home) / P["ball_scale_m"]), -4, 4)
    span = P["far_ball_m"] - P["near_ball_m"]
    w_ball = P["w_ball_far"] + (P["w_ball_near"] - P["w_ball_far"]) * np.clip(
        (P["far_ball_m"] - np.nan_to_num(dmin, nan=99.0)) / max(span, 1e-6), 0, 1)
    w_ball = np.where(have_ball, w_ball, 0.0)
    l_shape = np.clip(np.log(p_shape / (1 - p_shape)), -3, 3)
    team_logit = w_ball * l_ball + P["w_shape"] * l_shape + tag_team

    # S1: carry-forward of the last confident control (persistent kp frames), per block
    if C["lastctrl_w"]:
        dh = np.nan_to_num(d_home, nan=99.0)
        da = np.nan_to_num(d_away, nan=99.0)
        ctrl = np.full(n, -1)
        ctrl[raw_seen & (dh < ctrl_r) & (da > ctrl_margin)] = 0
        ctrl[raw_seen & (da < ctrl_r) & (dh > ctrl_margin)] = 1
        kp = int(C["ctrl_persist"])
        if kp > 1:
            keep = np.full(n, -1)
            for v in (0, 1):
                m = ctrl == v
                for s, e in zip(*runs_of(m)):
                    if e - s >= kp:
                        keep[s + kp - 1:e] = v
            ctrl = keep
        sgn = np.zeros(n)
        ar = np.arange(n)
        for _, first, last in blocks:
            c = ctrl[first:last + 1]
            ii = np.maximum.accumulate(np.where(c >= 0, ar[:len(c)], -1))
            st = np.where(ii >= 0, c[np.maximum(ii, 0)], -1)
            sgn[first:last + 1] = np.where(st == 0, 1.0, np.where(st == 1, -1.0, 0.0))
        team_logit = team_logit + C["lastctrl_w"] * sgn

    if C["w_lde"] or C["w_lde_dead"]:
        lpd, lph = _lde_logits(players, slot_team, ball_raw, sign_half1, blocks, pitch, FPS,
                               C["lde_clip"])
        team_logit = team_logit + C["w_lde"] * lph
        if C["w_lde_dead"]:
            p_dead = sigmoid(dead_logit + C["w_lde_dead"] * lpd)

    p_home = sigmoid(team_logit)
    eps = 1e-4
    pd = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    E = np.stack([np.log(1 - pd) + np.log(ph), np.log(1 - pd) + np.log(1 - ph), np.log(pd)], 1)
    aux = dict(d_home=d_home, d_away=d_away, out=out, seen=seen, raw_seen=raw_seen)
    extra = dict(gaps=gaps, kick_restarts=kick_restarts, k=k, P=P)
    return E * P["temperature"], aux, extra


# --------------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------------

def snap_to_gaps(path, aux, gaps, C, fps):
    """D end -> end of a long ball gap within the window; D start -> a long-gap start within the
    window, only if no ball-out frame exists in the look-back (then the out snap decides)."""
    path = path.copy()
    if not gaps:
        return path
    w = int(C["snap_win_s"] * fps)
    lb = int(C["out_lookback_s"] * fps)
    gs = np.array([g[0] for g in gaps])
    ge = np.array([g[1] for g in gaps])
    out = aux["out"]
    for s, e, lab in segments(path):
        if lab != D:
            continue
        if C["snap_gap_start"] and s > 0 and not out[max(0, s - lb):s + fps].any():
            j = int(np.argmin(np.abs(gs - s)))
            if abs(gs[j] - s) <= w and gs[j] < e:
                t = int(gs[j])
                if t < s:
                    path[t:s] = D
                elif t > s:
                    path[s:t] = path[s - 1]
        if C["snap_gap_end"] and e < len(path):
            j = int(np.argmin(np.abs(ge - e)))
            if abs(ge[j] - e) <= w and ge[j] > s:
                t = int(ge[j])
                if t > e:
                    path[e:t] = D
                elif t < e:
                    path[t:e] = path[e]
    return path


def snap_dead_start_back(path, aux, C, fps):
    """Each alive->D change: start D at the first ball-out frame in the previous look-back
    (staying keep_s after the possession's start). No fallback shift."""
    path = path.copy()
    segs = segments(path)
    out = aux["out"]
    lb, keep = int(C["out_lookback_s"] * fps), int(C["keep_s"] * fps)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, _, l = segs[i]
        if l != D or l0 == D:
            continue
        lo = max(s0 + keep, s - lb)
        if lo >= s:
            continue
        hits = np.where(out[lo:s + fps])[0]
        if len(hits):
            t = lo + int(hits[0])
            if t < s:
                path[t:s] = D
    return path


def merge_short_v2(path, P, fps):
    """Recompute segments after every merge, shortest offending segment first. prev == next ->
    that label; short alive between H and A -> the next team; short alive with exactly one D
    neighbour -> its alive neighbour; D-x-D kept."""
    min_len = {H: int(P["min_poss_s"] * fps), A: int(P["min_poss_s"] * fps),
               D: int(P["min_dead_s"] * fps)}
    path = path.copy()
    while True:
        segs = segments(path)
        if len(segs) == 1:
            break
        cand = sorted((e - s, i) for i, (s, e, lab) in enumerate(segs) if e - s < min_len[lab])
        done = False
        for _, i in cand:
            s, e, lab = segs[i]
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                path[s:e] = prev
            elif lab == D or prev is None or nxt is None:
                continue
            elif prev != D and nxt != D:
                path[s:e] = nxt
            else:                                   # exactly one D neighbour
                path[s:e] = nxt if prev == D else prev
            done = True
            break
        if not done:
            break
    return path


def prune_uncontrolled(path, aux, r, rounds):
    """Merge every H/A spell in which its team never controls the seen raw ball."""
    path = path.copy()
    seen = aux["raw_seen"]
    dh = np.nan_to_num(aux["d_home"], nan=99.0)
    da = np.nan_to_num(aux["d_away"], nan=99.0)
    for _ in range(rounds):
        segs = segments(path)
        changed = False
        for i, (a, b, lab) in enumerate(segs):
            if lab == D:
                continue
            own, oth = (dh, da) if lab == H else (da, dh)
            if (seen[a:b] & (own[a:b] < r) & (own[a:b] < oth[a:b])).any():
                continue
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                new = prev
            elif prev in (H, A) and nxt in (H, A):
                new = nxt
            elif prev in (H, A):
                new = prev
            elif nxt in (H, A):
                new = nxt
            else:
                continue
            if new != lab:
                path[a:b] = new
                changed = True
        if not changed:
            break
    return path


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps, cfg=None):
    C = dict(CFG, **(cfg or {}))
    P0 = dict(pp.PARAMS, **C["pp_over"])
    E, aux, extra = build_evidence(players, ball, slot_team, sign_half1, blocks, tags, names,
                                   pitch, P0, C)
    P = extra["P"]                                # radii scaled by k
    out = np.full(len(ball), 3, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        sub = {key: v[seg] for key, v in aux.items()}
        path = pp.viterbi(E[seg], P)
        if C["v1_merge"]:
            path = pp.merge_short(path, dict(P, min_poss_s=P["min_poss_s"] * C["v1_merge"]))
        if C["snap_v1"]:
            path = pp.snap_boundaries(path, sub, P)
        if C["snap_gap"]:
            g = [(s - first, e - first) for s, e in extra["gaps"] if s >= first and e <= last + 1]
            path = snap_to_gaps(path, sub, g, C, fps)
        if C["snap_back"]:
            path = snap_dead_start_back(path, sub, C, fps)
        if C["hygiene"]:
            path = merge_short_v2(path, P, fps)
            if C["prune"]:
                path = prune_uncontrolled(path, sub, P["control_radius_m"], int(C["prune_rounds"]))
                path = merge_short_v2(path, P, fps)
        if C["tag_fix"]:
            for s, rs in extra["kick_restarts"]:
                if (first - C["prekick_tol_s"] * fps <= s < first + C["prekick_s"] * fps
                        and rs > first):
                    path[:min(rs, last + 1) - first] = D
                    break
        out[seg] = path
    return out
