"""selftrain_ctrl core: ensemble_ctrl plus a model self-trained on the scored game's own pseudo-labels.

Per game, inside predict():
  1. Run the ensemble_vote members (temporal_tcn, handover_mstcn, ball_radar; under bench --dfl the
     fold rule and fold models of the game, exactly like ensemble_vote) and take their hard vote
     with ensemble_vote's smoothing -> codes ev and per-frame vote counts.
  2. Pseudo-labels: graded frames where every member votes the same label AND the frame is at least
     MIN_DIST frames from a label change of ev. Every STRIDE-th such frame is a training row.
  3. Features: learned_dfl_evidence's features.compute (copied here) on this game only. A small
     classifier (LightGBM, or a single decision tree) is fitted on the pseudo-labels plus their
     home/away-mirrored copy (feature swap_index, labels H <-> A), deterministic (1 thread, fixed
     seed). Its P(H, A, D) is predicted for every graded frame and symmetrised.
  4. Decode: argmax(vote counts + rank tie-break + W * P_selftrained), ensemble_vote's
     merge_short_v2 smoothing, then control.fold at CONTROL_S = 2.0 s (ensemble_ctrl's rule).

No label of the scored game is used: the pseudo-labels are the ensemble's own output (algorithmic).
The classifier is fitted fresh on every game and never saved, so there is no DFL-trained artefact
beyond the members' own (mapped through their folds). The settings (MIN_DIST, STRIDE, W, model
parameters) were chosen on DFL leave-one-game-out (bench --dfl2, ctrl_mean).
"""

import importlib.util
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (CAND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import control          # noqa: E402
import dfl_folds        # noqa: E402


def _private(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fe = _private("selftrain_ctrl_features", "features.py")
stm = _private("selftrain_ctrl_st_model", "st_model.py")
ec = _private("selftrain_ctrl_ensemble_core", "ensemble_core.py")
members = _private("selftrain_ctrl_members", "members.py")

H, A, D, X = 0, 1, 2, 3
CONTROL_S = 2.0


class SelfTrain(object):
    def __init__(self, model="lgb", params=None, min_dist=12, stride=5, w=1.5):
        with open(dfl_folds.resolve(os.path.join(HERE, "config_hard.json"))) as handle:
            self.cfg = json.load(handle)
        assert self.cfg["kind"] == "hard"
        self.slugs = list(self.cfg["slugs"])
        self.weights = self.cfg.get("weights")
        self.members = {s: members.Member(s) for s in self.slugs}
        self.lde = (self.members["learned_dfl_evidence"].mod if "learned_dfl_evidence" in self.members
                    else members.load("learned_dfl_evidence"))
        self.model, self.params = model, dict(params or {})
        self.min_dist, self.stride, self.w = int(min_dist), int(stride), float(w)
        self._cache = {}
        self.last_info = {}

    def _smooth(self, lab, blocks, fps):
        sm = self.cfg.get("smooth")
        out = np.full(len(lab), X, np.int8)
        for _, first, last in blocks:
            path = lab[first:last + 1].astype(np.int8).copy()
            if sm:
                path = self.lde.merge_short_v2(path, dict(min_poss_s=sm["min_poss_s"],
                                                          min_dead_s=sm["min_dead_s"]), fps)
            out[first:last + 1] = path
        return out

    def _run(self, args):
        key = members.input_key(args)
        if key in self._cache:
            return self._cache[key]
        self._cache.clear()
        players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps = args
        fps = int(fps)
        n = len(ball)
        mask = ec.block_mask(blocks, n)
        outs = {s: self.members[s].run(args) for s in self.slugs}
        codes_list = [outs[s][0] for s in self.slugs]
        ev = ec.hard_vote(codes_list, blocks, self.lde, fps, self.weights, self.cfg.get("smooth"))
        K = len(self.slugs)
        w = np.ones(K) if self.weights is None else np.asarray(self.weights, float)
        counts = ec.vote_shares(codes_list, w)                          # weighted label votes
        tie = ec.vote_shares(codes_list, ec.EPS_RANK * (K - np.arange(K)))   # hard_vote's tie-break
        # pseudo-labels on this game
        conf = stm.pseudo_labels(counts, float(w.sum()), ev, blocks, mask, self.min_dist, True)
        idx = np.where(conf)[0][::self.stride]
        sign = fe.frame_sign(n, sign_half1, blocks)
        F, fnames, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        perm, signs = fe.swap_index(fnames, kinds)
        P_st = stm.fit_predict(F, np.asarray(ev), idx, perm, signs, self.model, self.params, True, mask)
        S = counts + tie + self.w * P_st
        lab = np.argmax(S, 1).astype(np.int8)
        lab[~mask] = X
        codes = self._smooth(lab, blocks, fps)
        codes = control.fold(codes, CONTROL_S, [(int(a), int(b) + 1) for _, a, b in blocks], fps)
        P = (counts + self.w * P_st) / (float(w.sum()) + self.w)
        P = np.where(mask[:, None], P, 1.0 / 3)
        self.last_info = dict(n_train=int(len(idx)), conf_share=float(conf[mask].mean()))
        self._cache[key] = (np.asarray(codes, np.int8), P)
        return self._cache[key]

    def predict(self, *args):
        return self._run(args)[0]

    def predict_proba(self, *args):
        return self._run(args)[1]
