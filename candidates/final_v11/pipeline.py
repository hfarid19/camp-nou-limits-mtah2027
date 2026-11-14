"""final_v11 (submission CANDIDATE v11; docs/prereg/round5_prereg.txt addendum of 13 Nov 2026 18:32 CST): v9's member set
with the TCN and the MS-TCN replaced by their retrained "+ACC+SHAPE" versions (223 inputs = the repo features + 29 new
columns, candidates/acc_shape), plus a Metrica-trained stacker (S2 recipe, retrained on these members) as a FIFTH member of
the same MBR decode. Radar, pointer, decoder config and post-processing are v9's:

    TCN slot      acc_shape/member_tcn_as_tta4_a50.py       tcn_as.npz + test-time averaging (4 jittered copies) + all_a50
    MS-TCN slot   acc_shape/member_mstcn_as_skcft_metft.py  mstcn_as_skcft_metft.npz
    radar slot    ball_radar_pff/pipeline.py                radar_pff
    pointer slot  nn_pointer/pipeline_ballc_p20m.py         r5_ptr_ballc_p20M (19.3M-parameter ball-centred pointer)
    stacker       stacker_v11.txt (this folder)             stacker_met/stacker.py's predict on the four members' float32 probs

Decoder config: src hmm, alpha 0.1, rho 0.2, margin 0.0; fixed sampling seed (decoder.SEED). Order of operations as
candidates/final_f4sps/pipeline.py (v10): member probabilities rounded to float32, stacker at every frame inside the graded
blocks (one thread), its probabilities rounded to float32, decoder.decode(four members + stacker), then prekick_dead.
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

MEMBERS = [("acc_shape", "member_tcn_as_tta4_a50.py"),
           ("acc_shape", "member_mstcn_as_skcft_metft.py"),
           ("ball_radar_pff", "pipeline.py"),
           ("nn_pointer", "pipeline_ballc_p20m.py")]
CFG = dict(src="hmm", alpha=0.1, rho=0.2, margin=0.0)
MODEL = os.path.join(HERE, "stacker_v11.txt")
META = os.path.join(HERE, "stacker_v11_meta.json")


def load_stacker():
    """A fresh Booster of stacker_v11.txt (LightGBM version and file sha256 checked against stacker_v11_meta.json)."""
    import lightgbm as lgb
    with open(META) as handle:
        meta = json.load(handle)
    assert lgb.__version__ == meta["lgb_version"], "lightgbm version differs from training"
    with open(MODEL, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == meta["sha256"], "stacker_v11.txt differs from its meta"
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

    def run(self, args):
        key = members.input_key(args)
        if key in self.cache:
            return self.cache[key]
        self.cache.clear()
        blocks, n = args[4], len(args[2])
        codes_list, probs = [], []
        for m in self.members:
            codes_list.append(np.asarray(m.mod.predict(*args), np.int8))
            probs.append(np.asarray(m.mod.predict_proba(*args), np.float32).astype(np.float64))
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
