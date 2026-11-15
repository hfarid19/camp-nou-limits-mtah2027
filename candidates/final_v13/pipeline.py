"""final_v13 (CANDIDATE v13; docs/prereg/round5_prereg.txt, DECISION of 14 Nov 2026 16:20 CST): final_v12 with unequal member
weights in the MBR decode. Nothing is retrained: the five members, the stacker file (candidates/final_v12/stacker_v12.txt),
the decoder config and the post-processing are v12's. Only the list handed to the decoder changes:

    weights (TCN, MS-TCN, radar, pointer, stacker) = (1, 3, 1, 2, 3)

A weight w means the member appears w times in the decoder's list (probabilities and codes), as in the member-weight search
of 14 Nov 2026 (weights 0..3 per member, 1023 weightings). List order = "rounds": the members with weight >= 1, then those
with weight >= 2, then those with weight >= 3, each round in v12 order. So the list is

    TCN, MS-TCN, radar, pointer, stacker,  MS-TCN, pointer, stacker,  MS-TCN, stacker        (10 entries)

The weighting was chosen by the team after seeing the search's results and it did NOT pass the rule written before the
search (see README.md in this folder). The stacker still reads the four members once each; its inputs do not change.
predict_proba = vote shares of the five members' codes, unweighted (not used by make_submission.py).
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
V12 = os.path.join(CAND, "final_v12")
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), os.path.join(CAND, "mbr_decode"), os.path.join(CAND, "stacker_met"),
           HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402
import v6_core              # noqa: E402
import decoder              # noqa: E402
import stacker              # noqa: E402  (feats / predict only; the model file is final_v12's)

MEMBERS = [("newfeat_v12", "member_tcn_v12_tta4_a50.py"),
           ("newfeat_v12", "member_mstcn_v12_skcft_metft.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc_p20m.py")]
CFG = dict(src="hmm", alpha=0.1, rho=0.2, margin=0.0)
WEIGHTS = (1, 3, 1, 2, 3)   # TCN, MS-TCN, radar, pointer, stacker
MODEL = os.path.join(V12, "stacker_v12.txt")
META = os.path.join(V12, "stacker_v12_meta.json")


def weighted_order(weights=WEIGHTS):
    """Member indices in rounds order: [w >= 1] + [w >= 2] + [w >= 3], each in member order."""
    return [i for r in range(1, max(weights) + 1) for i in range(len(weights)) if weights[i] >= r]


def load_stacker():
    """A fresh Booster of final_v12's stacker_v12.txt (LightGBM version and file sha256 checked against its meta)."""
    import lightgbm as lgb
    with open(META) as handle:
        meta = json.load(handle)
    assert lgb.__version__ == meta["lgb_version"], "lightgbm version differs from training"
    with open(MODEL, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == meta["sha256"], "stacker_v12.txt differs from its meta"
    return lgb.Booster(model_file=MODEL)


class WeightedStackVote(object):
    """final_v12.pipeline.StackVote with a weighted member list for the decoder. self.last keeps the latest unit's
    member outputs (four members, then the stacker), once each."""

    def __init__(self, member_specs, cfg, weights):
        self.members = [members.Member(s, f) for s, f in member_specs]
        self.cfg = dict(cfg)
        self.order = weighted_order(weights)
        self.cache = {}
        self.last = None
        self.booster = None

    def _release(self):
        """Memory only (no effect on any output); see final_v12.pipeline.StackVote._release."""
        fe = sys.modules.get("mtah_newfeat_v12_fe")
        if fe is not None:
            fe._CACHE.clear()
        for m in self.members[:2]:
            inner = getattr(m.mod, "_mod", None)
            if inner is not None and isinstance(getattr(inner, "_CACHE", None), dict):
                inner._CACHE.clear()

    def run(self, args):
        key = members.input_key(args)
        if key in self.cache:
            return self.cache[key]
        self.cache.clear()
        blocks, n = args[4], len(args[2])
        codes_list, probs = [], []
        for i, m in enumerate(self.members):
            codes_list.append(np.asarray(m.mod.predict(*args), np.int8))
            probs.append(np.asarray(m.mod.predict_proba(*args), np.float32).astype(np.float64))
            if i == 1:
                self._release()
        if self.booster is None:
            self.booster = load_stacker()
        sp, sc = stacker.predict(probs, blocks, self.booster)
        codes_list.append(sc)
        probs.append(np.asarray(sp, np.float32).astype(np.float64))
        self.last = (list(codes_list), list(probs))
        codes, _ = decoder.decode([probs[i] for i in self.order], [codes_list[i] for i in self.order], blocks, self.cfg)
        codes = v6_core.prekick_dead(np.asarray(codes, np.int8), args[5], blocks)
        mask = ec.block_mask(blocks, n)
        P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
        self.cache[key] = (np.asarray(codes, np.int8), P)
        return self.cache[key]


_VOTE = WeightedStackVote(MEMBERS, CFG, WEIGHTS)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
