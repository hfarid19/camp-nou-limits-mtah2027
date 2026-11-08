"""learned_dfl_evidence: LightGBM H/A/D frame evidence learned on open DFL tracking + Viterbi.

README
------
Evidence model. Two LightGBM boosters (model_d.txt: dead vs alive; model_ha.txt: home vs away
while alive) on ~190 identity-free, team-relative per-frame features (features.py: ball gap
context, nearest-player distances / counts per team, team shape, last confident control,
restart patterns, multi-scale temporal context). They are trained ONLY on externally labelled
tracking of OTHER games: the open DFL / IDSSE dataset (7 Bundesliga matches, per-frame
BallPossession and BallStatus flags from the tracking), degraded with domain randomisation to
look like the competition's automatic tracking (make_dataset.py). Nothing is trained on Metrica
or on the competition match; the bench only picks the 4 decoder scalars below.

Data source and licence. Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "An integrated
dataset of spatiotemporal and event data in elite soccer." Scientific Data 12. figshare article
28196177, https://doi.org/10.6084/m9.figshare.28196177 . Data (c) DFL Deutsche Fussball Liga /
IDSSE, licensed CC BY 4.0. Downloaded by fetch_idsse.py (raw data is cached outside the repo and
never committed; see .gitignore).

Rebuild:  python -I fetch_idsse.py --games 7 ; python -I make_dataset.py ; python -I train.py
New dependency: lightgbm (pip install lightgbm, 4.7.0 used), deterministic with num_threads=1.

A third booster (model_c.txt) is a hand-over detector: P(a direct H<->A change of the
tracking possession flag within +-4 frames), trained on alive frames with swap augmentation.
Rounds of all three picked by leave-one-DFL-game-out log-loss (d 523, ha 389, c 353 rounds;
LOGO log-loss 0.058 / 0.273 / 0.122). Inference averages each prediction with the prediction on
the team-swapped features, so H/A are exactly symmetric.

DFL label diagnostics (7 games, BallStatus/BallPossession): 412 possession spells per 90 min,
dead share 0.448; spell duration p10/p25/p50/p75/p90 = 0.8/1.7/3.7/8.5/18.6 s (12.5% < 1 s);
stoppage duration p10/p50/p90 = 5.8/19.6/49.4 s. BallPossession keeps the last team while the
ball is in flight and flips on touches, so it has more short spells than event-derived labels.

Decoder. E[:,H] = tau*(log pH + log(1-pD)), E[:,A] = tau*(log(1-pH) + log(1-pD)),
E[:,D] = tau*log pD; possession_pipeline.viterbi with switch / dead costs, merge_short,
H<->A boundaries snapped to the hand-over detector's peak within 1.5 s (snap_mode 1; the
first-hit rule of possession_pipeline.snap_boundaries is snap_mode 0), then gap-edge D snaps
(a D end moved to the end of the ball gap it sits in, within 4 s). Smart Tagging priors (match
only, a no-op on the bench) are kept small: goals -> dead window, build up / progression team
prior at w 0.2 (changes 538 match frames, 0.3%).

Selection (dev bench only): tau is a no-op because the costs scale with it (Viterbi is scale
invariant), so the choice was switch_cost x dead_cost x beta (x gamma). Plateau: switch
0.5-1.0, dead 4-6 all within 0.611-0.613; picked the centre 0.75 / 5. beta 0.3 (blend v1's
in-match shape logit) cost 0.008 spellF1, so beta 0. gamma (hand-over detector on the switch
cost) was +-0.001 on the bench and on DFL, so 0 (the detector is still used for snapping).

Results. Dev bench spellF1 0.6129, macroF1 0.7988 (v1 0.4749 / 0.7453), every row +0.12..0.15.
Ablation on the same decoder (switch 0.75): learned dead + v1 team 0.5226; v1 dead + learned
team 0.5613; v1 both 0.4856; learned both 0.6129. DFL leave-one-game-out (degraded with an
unseen seed 21, fold models, tracking-flag labels): spellF1 0.527, macroF1 0.893 (robustness
diagnostic only). Real match: 317 possessions, 43.7% dead. Two --match runs are byte-identical.
"""

import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import possession_pipeline as pp  # noqa: E402
import features as fe             # noqa: E402

warnings.filterwarnings("ignore")
H, A, D = 0, 1, 2

CFG = dict(
    tau=0.15,             # evidence temperature
    switch_cost=0.75,     # H <-> A (at tau 0.15; scaled with tau). Bench plateau 0.5-1.0
    dead_cost=5.0,        # alive <-> D (at tau 0.15; scaled with tau)
    beta=0.0,             # blend weight of v1's in-match team-shape logit
    gamma=0.0,            # weight of the learned hand-over detector on the switch cost
    snap_mode=1.0,        # 0: first-hit snap of H<->A changes, 1: snap to the hand-over peak
    min_poss_s=1.0,
    min_dead_s=1.0,
    refine_window_s=1.5,
    control_radius_m=1.5,
    d_end_snap_s=4.0,     # move a D end to the end of the ball gap it sits in
    d_start_lookback_s=6.0,  # move a D start back to the first out-of-pitch frame
    w_tag_team=0.2,
    w_tag_goal=1.0,
)
for _k in list(CFG):
    if "LDE_" + _k.upper() in os.environ:
        CFG[_k] = float(os.environ["LDE_" + _k.upper()])

_MODELS = {}


def _models():
    if not _MODELS:
        import lightgbm as lgb
        _MODELS["d"] = lgb.Booster(model_file=os.path.join(HERE, "model_d.txt"))
        _MODELS["ha"] = lgb.Booster(model_file=os.path.join(HERE, "model_ha.txt"))
        cpath = os.path.join(HERE, "model_c.txt")
        _MODELS["c"] = lgb.Booster(model_file=cpath) if os.path.isfile(cpath) else None
        with open(os.path.join(HERE, "model_meta.json")) as handle:
            _MODELS["meta"] = json.load(handle)
    return _MODELS


def frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps, models=None):
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    m = models or _models()
    assert names == m["meta"]["names"], "feature list differs from the trained model"
    p_dead = m["d"].predict(F, num_threads=1)
    perm, signs = fe.swap_index(names, kinds)
    p1 = m["ha"].predict(F, num_threads=1)
    p2 = m["ha"].predict(F[:, perm] * signs, num_threads=1)
    p_home = 0.5 * (p1 + 1.0 - p2)                     # exact H/A symmetry
    aux = dict(F=F, names=names, p_change=None)
    if m.get("c") is not None:
        aux["p_change"] = 0.5 * (m["c"].predict(F, num_threads=1)
                                 + m["c"].predict(F[:, perm] * signs, num_threads=1))
        aux["change_prior"] = m["meta"].get("change_prior", 0.04)
    return p_dead, p_home, aux


def tag_priors(tags, names, n, fps):
    team = np.zeros(n)
    dead = np.zeros(n)
    home_name, away_name = names[0].lower(), names[1].lower()

    def side(t):
        if not t or t in ("n/a", "na"):
            return 0.0
        if t == home_name or "barc" in t:
            return 1.0
        if t == away_name or "atl" in t or "athl" in t:
            return -1.0
        return 0.0
    for code, s, e, t, _kind in tags:
        s, e = max(0, s), min(n, max(e, s + 1))
        if code in ("build up", "progression"):
            team[s:e] += side(t)
        elif code == "goals":
            dead[e:min(n, e + 10 * fps)] = 1.0
    return np.clip(team, -1, 1), dead


def viterbi_tv(E, sw, din, dout):
    """3-state Viterbi (H, A, D) with a per-frame H <-> A switch cost sw[t]."""
    T = len(E)
    back = np.zeros((T, 3), np.int8)
    e = E.tolist()
    sw = sw.tolist()
    sH, sA, sD = e[0]
    for t in range(1, T):
        c = sw[t]
        eh, ea, ed = e[t]
        # into H
        a, b, d = sH, sA - c, sD - dout
        if a >= b and a >= d:
            nH, bh = a, 0
        elif b >= d:
            nH, bh = b, 1
        else:
            nH, bh = d, 2
        a, b, d = sH - c, sA, sD - dout
        if b >= a and b >= d:
            nA, ba = b, 1
        elif a >= d:
            nA, ba = a, 0
        else:
            nA, ba = d, 2
        a, b, d = sH - din, sA - din, sD
        if d >= a and d >= b:
            nD, bd = d, 2
        elif a >= b:
            nD, bd = a, 0
        else:
            nD, bd = b, 1
        back[t] = (bh, ba, bd)
        sH, sA, sD = nH + eh, nA + ea, nD + ed
    path = np.empty(T, np.int8)
    path[-1] = int(np.argmax([sH, sA, sD]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_to_peak(path, pc, fps, window_s):
    """Move each H <-> A change to the frame of highest hand-over probability nearby."""
    path = path.copy()
    w = int(window_s * fps)
    segs = pp.segments(path)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, e, l = segs[i]
        if l0 == D or l == D:
            continue
        lo, hi = max(s0 + 1, s - w), min(e - 1, s + w)
        if hi <= lo:
            continue
        t = lo + int(np.argmax(pc[lo:hi]))
        path[min(t, s):max(t, s)] = l if t < s else l0
    return path


def snap_dead_edges(path, out, raw_seen, fps, end_s, look_s):
    path = path.copy()
    segs = pp.segments(path)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, e, l = segs[i]
        if l == D and l0 != D and look_s > 0:          # D start back to the first out frame
            lo = max(s0 + fps // 2, s - int(look_s * fps))
            hits = np.where(out[lo:s])[0]
            if len(hits):
                path[lo + int(hits[0]):s] = D
        if l0 == D and l != D and end_s > 0:           # D end forward to the gap's end
            if not raw_seen[s]:
                k = s
                lim = min(e - 1, s + int(end_s * fps))
                while k < lim and not raw_seen[k]:
                    k += 1
                if k < lim:
                    path[s:k] = D
    return path


def decode(p_dead, p_home, aux, blocks, n, fps, tags=(), names=("Home", "Away"), cfg=None,
           shape_logit=None):
    c = dict(CFG, **(cfg or {}))
    tau = c["tau"]
    pdd = np.clip(p_dead, 1e-3, 1 - 1e-3)
    ph = np.clip(p_home, 1e-3, 1 - 1e-3)
    lh = np.log(ph / (1 - ph))
    if shape_logit is not None and c["beta"] > 0:
        lh = (1 - c["beta"]) * lh + c["beta"] * shape_logit
    ld = np.log(pdd / (1 - pdd))
    if len(tags):
        tt, td = tag_priors(tags, names, n, fps)
        lh = lh + c["w_tag_team"] * 2.0 * tt
        ld = ld + c["w_tag_goal"] * td
    ph = 1 / (1 + np.exp(-lh))
    pdd = 1 / (1 + np.exp(-ld))
    ph = np.clip(ph, 1e-3, 1 - 1e-3)
    pdd = np.clip(pdd, 1e-3, 1 - 1e-3)
    E = tau * np.stack([np.log(ph) + np.log(1 - pdd), np.log(1 - ph) + np.log(1 - pdd),
                        np.log(pdd)], 1)
    scale = tau / 0.15
    P = dict(pp.PARAMS, switch_cost=c["switch_cost"] * scale,
             dead_in_cost=c["dead_cost"] * scale, dead_out_cost=c["dead_cost"] * scale,
             min_poss_s=c["min_poss_s"], min_dead_s=c["min_dead_s"],
             refine_window_s=c["refine_window_s"], control_radius_m=c["control_radius_m"])
    F, nm = aux["F"], aux["names"]
    col = {k: i for i, k in enumerate(nm)}
    d_home = F[:, col["H_d1"]].astype(np.float64)
    d_away = F[:, col["A_d1"]].astype(np.float64)
    seen = ~np.isnan(d_home) | ~np.isnan(d_away)
    raw_seen = F[:, col["ball_seen"]] > 0.5
    out = np.nan_to_num(F[:, col["ball_out"]], nan=0.0) > 0.5
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    pc = aux.get("p_change")
    use_c = pc is not None and c["gamma"] > 0
    if use_c:
        pcc = np.clip(pc, 1e-3, 1 - 1e-3)
        prior = aux["change_prior"]
        bonus = np.log(pcc / (1 - pcc)) - np.log(prior / (1 - prior))
        sw = np.maximum(0.02, P["switch_cost"] - c["gamma"] * tau * bonus)
    pred = np.full(n, 3, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        if use_c:
            path = viterbi_tv(E[seg], sw[seg], P["dead_in_cost"], P["dead_out_cost"])
        else:
            path = pp.viterbi(E[seg], P)
        path = pp.merge_short(path, P)
        if pc is not None and c["snap_mode"] >= 0.5:
            path = snap_to_peak(path, pc[seg], fps, c["refine_window_s"])
        else:
            path = pp.snap_boundaries(path, dict(d_home=d_home[seg], d_away=d_away[seg],
                                                 out=out[seg], seen=seen[seg]), P)
        path = snap_dead_edges(path, out[seg], raw_seen[seg], fps, c["d_end_snap_s"],
                               c["d_start_lookback_s"])
        pred[seg] = path
    return pred


def v1_shape_logit(players, slot_team, ball, sign_half1, blocks, tags, names, pitch):
    """v1's in-match pseudo-label team-shape logit (self-generated labels), for the beta blend."""
    P = dict(pp.PARAMS, temperature=1.0, w_ball_near=0.0, w_ball_far=0.0, w_tag=0.0,
             w_shape=1.0)
    E, _ = pp.build_evidence(players, ball, slot_team, sign_half1, blocks, [], names, pitch, P,
                             lambda *a, **k: None)
    return np.clip(E[:, 0] - E[:, 1], -3, 3)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    p_dead, p_home, aux = frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    shape = None
    if CFG["beta"] > 0:
        shape = v1_shape_logit(players, slot_team, ball, sign_half1, blocks, tags, names, pitch)
    return decode(p_dead, p_home, aux, blocks, len(ball), fps, tags, names, shape_logit=shape)
