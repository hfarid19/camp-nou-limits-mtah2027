"""Load other candidates' pipelines side by side in one process and get (codes, P(H,A,D)) from each.

Every candidate folder has its own `features.py`, `lde_core.py`, `tcn_np.py`, ... with the same
module names, so a plain import of two of them would silently share the first one's modules. load()
therefore executes each member's pipeline.py under a private module name with only that member's
folder at the front of sys.path, after removing any module of the same name from sys.modules, and
afterwards removes the member's own modules from sys.modules again (the member keeps its references
in its globals). sys.path is restored after each load. No member imports a local module lazily
(checked: only stdlib, numpy, scipy, lightgbm, possession_pipeline and dfl_folds), so this isolation
holds at predict time too.

Members are loaded fresh every time this module is executed. bench.py --dfl re-executes the
candidate (and so this file) for every DFL game and swaps lightgbm.Booster for the fold model of that
game, so each member's lazily cached boosters are the fold models of the game being scored; every
other DFL-trained file goes through the members' own dfl_folds.resolve() calls.

The three older candidates (learned_dfl_evidence, v2_stacked_fixes, hier_touch_decoder) have no
predict_proba. Wrappers below compute it from the evidence their own decoders work from, sharing one
feature pass with predict:
  learned_dfl_evidence  P(D) = p_dead, P(H) = (1 - p_dead) p_home after the decoder's own long-gap
                        and local-contrast adjustments (the exact inputs of its Viterbi)
  v2_stacked_fixes      exp(E / temperature) of build_evidence (its rows are already (1-pd)ph,
                        (1-pd)(1-ph), pd)
  hier_touch_decoder    P(D) = sigmoid(stage-A dead logit), P(H | alive) = sigmoid(stage-B base team
                        logit, after its local contrast)
"""

import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True          # never write __pycache__ into other candidates' folders

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)

ALL = ["learned_dfl_evidence", "v2_stacked_fixes", "hier_touch_decoder", "pathcrf_dynamic",
       "ball_radar", "temporal_tcn", "handover_mstcn", "semimarkov_lattice", "ballfree_dead",
       "pitch_control", "skillcorner_domain"]
WRAPPED = {"learned_dfl_evidence", "v2_stacked_fixes", "hier_touch_decoder"}


def _local_names(folder):
    return {f[:-3] for f in os.listdir(folder) if f.endswith(".py")}


def load(slug, fname="pipeline.py"):
    """Execute candidates/<slug>/<fname> in isolation; returns the module. fname selects a
    training-target variant (e.g. pipeline_raw.py)."""
    folder = os.path.join(CAND, slug)
    if not os.path.isdir(folder):                # members not shipped live in candidates/archive/
        folder = os.path.join(CAND, "archive", slug)
    local = _local_names(folder)
    saved_path = list(sys.path)
    for name in local:
        sys.modules.pop(name, None)
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location("ensv_member_" + slug + "_" + fname[:-3],
                                                      os.path.join(folder, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name in local:
            m = sys.modules.get(name)
            if m is not None and os.path.abspath(getattr(m, "__file__", "") or "").startswith(
                    folder + os.sep):
                del sys.modules[name]
        sys.path[:] = saved_path
    return mod


def _norm(p):
    p = np.nan_to_num(np.asarray(p, np.float64), nan=1.0 / 3)
    p = np.clip(p, 0.0, None)
    s = p.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return p / s


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


class Member(object):
    def __init__(self, slug, fname="pipeline.py"):
        self.slug = slug
        self.mod = load(slug, fname)

    def run(self, args):
        """(codes int8 (N,), proba float64 (N, 3), side dict or None) for one game."""
        m = self.mod
        if self.slug == "learned_dfl_evidence":
            return _run_lde(m, args)
        if self.slug == "v2_stacked_fixes":
            return _run_v2(m, args)
        if self.slug == "hier_touch_decoder":
            return _run_htd(m, args)
        codes = np.asarray(m.predict(*args), np.int8)
        proba = _norm(m.predict_proba(*args))
        return codes, proba, None


def lde_evidence_probs(p_dead, p_home, aux, tags, names, cfg, fps):
    """The (ph, pdd) learned_dfl_evidence.decode feeds its Viterbi, before tau (tags included)."""
    import possession_pipeline as pp
    c = cfg
    pdd = np.clip(p_dead, 1e-3, 1 - 1e-3)
    ph = np.clip(p_home, 1e-3, 1 - 1e-3)
    lh = np.log(ph / (1 - ph))
    if c["contrast_beta"] > 0:
        q = np.clip(p_home, 1e-6, 1 - 1e-6)
        l6 = np.log(q / (1 - q))
        ls = pp.moving_mean(l6, int(c["contrast_hs"])) if c["contrast_hs"] > 0 else l6
        cc = ls - pp.moving_mean(l6, int(c["contrast_hw"]))
        cc = np.sign(cc) * np.maximum(np.abs(cc) - c["contrast_gate"], 0.0)
        lh = np.clip(l6 + c["contrast_beta"] * cc, -np.log(999.0), np.log(999.0))
    ld = np.log(pdd / (1 - pdd))
    if aux.get("long_gap") is not None and c["w_gap"] != 0:
        ld = ld + c["w_gap"] * aux["long_gap"]
    F, nm = aux["F"], aux["names"]
    col = {k: i for i, k in enumerate(nm)}
    out = np.nan_to_num(F[:, col["ball_out"]], nan=0.0) > 0.5
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    if c["w_out"] != 0:
        ld = ld + c["w_out"] * out
    if len(tags):
        raw_seen = F[:, col["ball_seen"]] > 0.5
        tt, td = c["_tag_priors"](tags, names, len(ph), fps, raw_seen, c["goal_win_s"])
        lh = lh + c["w_tag_team"] * tt
        ld = ld + c["w_tag_goal"] * td
    ph = np.clip(_sig(lh), 1e-3, 1 - 1e-3)
    pdd = np.clip(_sig(ld), 1e-3, 1 - 1e-3)
    return ph, pdd


def _run_lde(m, args):
    players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps = args
    p_dead, p_home, aux = m.frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    aux["long_gap"] = m.long_gap_gated(players, ball, fps, m.CFG["gap_s"], m.CFG["gap_speed_ms"])
    aux["kick_restarts"] = m.kickoff_restarts(tags, blocks, ball, fps, m.CFG["prekick_s"])
    assert m.CFG["beta"] == 0
    codes = m.decode(p_dead, p_home, aux, blocks, len(ball), fps, tags, names)
    cfg = dict(m.CFG, _tag_priors=m.tag_priors)
    ph, pdd = lde_evidence_probs(p_dead, p_home, aux, tags, names, cfg, fps)
    proba = np.stack([(1 - pdd) * ph, (1 - pdd) * (1 - ph), pdd], 1)
    # decoder side-information the soft vote reuses (LDE's own snapping machinery)
    F, nm = aux["F"], aux["names"]
    col = {k: i for i, k in enumerate(nm)}
    side = dict(ball_seen=F[:, col["ball_seen"]].astype(np.float32),
                ball_out=F[:, col["ball_out"]].astype(np.float32),
                H_d1=F[:, col["H_d1"]].astype(np.float32), A_d1=F[:, col["A_d1"]].astype(np.float32),
                p_change=np.asarray(aux["p_change"], np.float32),
                long_gap=np.asarray(aux["long_gap"], np.float32),
                p_dead=np.asarray(p_dead, np.float32), p_home=np.asarray(p_home, np.float32))
    kr = aux["kick_restarts"]
    side["kick_first"] = np.array(sorted(kr), np.int64)
    side["kick_frame"] = np.array([kr[k] for k in sorted(kr)], np.int64)
    return np.asarray(codes, np.int8), _norm(proba), side


def _run_v2(m, args):
    players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps = args
    codes = np.asarray(m.predict(*args), np.int8)
    C = dict(m.CFG)
    P = dict(m.pp.PARAMS, **C["pp_over"])
    E, _, _ = m.build_evidence(players, ball, slot_team, sign_half1, blocks, tags, names, pitch,
                               P, C)
    proba = np.exp(E / P["temperature"])
    return codes, _norm(proba), None


def _run_htd(m, args):
    players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps = args
    C = m.CFG
    Fe = m.features(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    Fe = m.add_lde(Fe, players, slot_team, ball, sign_half1, blocks, pitch, fps, C)
    codes = np.asarray(m.decode(Fe, blocks, C), np.int8)
    Fc = dict(Fe)
    if C.get("lc_beta", 0) > 0 and Fc.get("lde_team") is not None:
        Fc["lde_team"] = m.local_contrast(np.asarray(Fc["lde_team"], np.float64), C)
    pdd = np.clip(_sig(m.dead_logit(Fc, C)), 1e-4, 1 - 1e-4)
    base = m.team_inputs(Fc, C)[0]
    ph = np.clip(_sig(base), 1e-4, 1 - 1e-4)
    proba = np.stack([(1 - pdd) * ph, (1 - pdd) * (1 - ph), pdd], 1)
    return codes, _norm(proba), None


def input_key(args):
    players, slot_team, ball = args[0], args[1], args[2]
    h = hashlib.sha256()
    for a in (players, slot_team, ball, np.asarray(args[4])):
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(repr((args[3], len(args[5]), tuple(args[6]), tuple(args[7]), args[8])).encode())
    return h.hexdigest()
