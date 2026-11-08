"""Live ensemble: runs the chosen member candidates on the game and combines them (ensemble_core).

The rule (members, weights, pooling, smoothing / decoder variant) is read from a JSON chosen on DFL
(tuning/fit.py, written by tuning/write_configs.py). That choice used DFL labels, so the file is
loaded through dfl_folds.resolve(): under bench --dfl, game gi gets fold_<kind>_<gi>.json, the rule
selected on the OTHER six games only (the nested leave-one-game-out choice), and every member's own
models are swapped for their fold models by the bench / the members' own resolve() calls.
"""

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
import dfl_folds        # noqa: E402
import members          # noqa: E402
import ensemble_core as ec  # noqa: E402


class Vote(object):
    def __init__(self, cfg_path):
        with open(dfl_folds.resolve(cfg_path)) as handle:
            self.cfg = json.load(handle)
        self.kind = self.cfg["kind"]
        self.slugs = list(self.cfg["slugs"])             # best member first (tie-break order)
        self.weights = self.cfg.get("weights")
        need = list(self.slugs)
        if "learned_dfl_evidence" not in need and self.kind == "soft":
            need.append("learned_dfl_evidence")         # decoder side information (model_c etc.)
        self.members = {s: members.Member(s) for s in need}
        self.lde = (self.members["learned_dfl_evidence"].mod if "learned_dfl_evidence" in self.members
                    else members.load("learned_dfl_evidence"))
        self._cache = {}

    def _run(self, args):
        key = members.input_key(args)
        if key in self._cache:
            return self._cache[key]
        self._cache.clear()
        outs = {s: m.run(args) for s, m in self.members.items()}
        blocks, n, fps = args[4], len(args[2]), int(args[8])
        mask = ec.block_mask(blocks, n)
        if self.kind == "hard":
            codes_list = [outs[s][0] for s in self.slugs]
            codes = ec.hard_vote(codes_list, blocks, self.lde, fps, self.weights,
                                 self.cfg.get("smooth"))
            P = ec.vote_proba(codes_list, blocks, self.weights)
        else:
            P = ec.pool([outs[s][1] for s in self.slugs], self.weights, self.cfg["pool"])
            side = outs["learned_dfl_evidence"][2]
            codes = ec.soft_decode(P, side, blocks, n, fps, self.lde, self.cfg.get("decoder"))
        P = np.where(mask[:, None], P, 1.0 / 3)
        self._cache[key] = (np.asarray(codes, np.int8), P)
        return self._cache[key]

    def predict(self, *args):
        return self._run(args)[0]

    def predict_proba(self, *args):
        return self._run(args)[1]
