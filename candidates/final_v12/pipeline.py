"""final_v12 (CANDIDATE v12; docs/prereg/round5_prereg.txt addenda of 13 Nov 2026 20:52 and 14 Nov 2026 01:19 CST): final_v11
with the TCN and the MS-TCN replaced by their "v12" versions (281 inputs = the repo features + ACC + SHAPE + REL + KIN2 +
INT-hand columns, candidates/newfeat_v12), plus the S2-recipe stacker refitted on these members as a FIFTH member of the
same MBR decode. Radar, pointer, decoder config and post-processing are v9's:

    TCN slot      newfeat_v12/member_tcn_v12_tta4_a50.py       tcn_v12.npz + test-time averaging (4 jittered copies) + all_a50
    MS-TCN slot   newfeat_v12/member_mstcn_v12_skcft_metft.py  mstcn_v12_skcft_metft.npz
    radar slot    ball_radar_pff/pipeline.py                radar_pff
    pointer slot  nn_pointer/pipeline_ballc_p20m.py         r5_ptr_ballc_p20M (19.3M-parameter ball-centred pointer)
    stacker       stacker_v12.txt (this folder)             stacker_met/stacker.py's predict on the four members' float32 probs

Decoder config: src hmm, alpha 0.1, rho 0.2, margin 0.0; fixed sampling seed (decoder.SEED). Order of operations as
candidates/final_f4sps/pipeline.py (v10): member probabilities rounded to float32, stacker at every frame inside the graded
blocks (one thread), its probabilities rounded to float32, decoder.decode(four members + stacker), then prekick_dead.
Memory: after the MS-TCN member the feature-block cache and the two NN members' unit caches are released (outputs unchanged).
predict_proba = vote shares of the five members' codes (not used by make_submission.py).
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
for _p in (os.path.join(CAND, "ensemble_ctrl_v6"), os.path.join(CAND, "mbr_decode"), os.path.join(CAND, "stacker_met"),
           HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import members              # noqa: E402
import ensemble_core as ec  # noqa: E402
import v6_core              # noqa: E402
import decoder              # noqa: E402
import stacker              # noqa: E402  (feats / predict only; the model file is this folder's)

MEMBERS = [("newfeat_v12", "member_tcn_v12_tta4_a50.py"),
           ("newfeat_v12", "member_mstcn_v12_skcft_metft.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc_p20m.py")]
CFG = dict(src="hmm", alpha=0.1, rho=0.2, margin=0.0)
MODEL = os.path.join(HERE, "stacker_v12.txt")
META = os.path.join(HERE, "stacker_v12_meta.json")


def load_stacker():
    """A fresh Booster of stacker_v12.txt (LightGBM version and file sha256 checked against stacker_v12_meta.json)."""
    import lightgbm as lgb
    with open(META) as handle:
        meta = json.load(handle)
    assert lgb.__version__ == meta["lgb_version"], "lightgbm version differs from training"
    with open(MODEL, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == meta["sha256"], "stacker_v12.txt differs from its meta"
    return lgb.Booster(model_file=MODEL)


class StackVote(object):
    """final_f4sps.pipeline.StackVote with this folder's stacker. self.last keeps the latest unit's member outputs
    (four members, then the stacker)."""

    def __init__(self, member_specs, cfg):
        self.members = [members.Member(s, f) for s, f in member_specs]
        self.cfg = dict(cfg)
        self.cache = {}
        self.last = None
        self.booster = None

    def _release(self):
        """Memory only (no effect on any output): once the TCN and MS-TCN members have produced their codes and
        probabilities for this unit, drop the newfeat_v12 feature-block cache and the two members' per-unit caches before
        radar and pointer run (each member is called exactly once per unit; the caches only serve predict / predict_proba
        of the same member and unit)."""
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
        codes, _ = decoder.decode(probs, codes_list, blocks, self.cfg)
        codes = v6_core.prekick_dead(np.asarray(codes, np.int8), args[5], blocks)
        mask = ec.block_mask(blocks, n)
        P = np.where(mask[:, None], ec.vote_proba(codes_list, blocks), 1.0 / 3)
        self.cache[key] = (np.asarray(codes, np.int8), P)
        return self.cache[key]


_VOTE = StackVote(MEMBERS, CFG)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _VOTE.run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
