"""Offline scoring of vote rules on the members' collected DFL / dev outputs (collect/<member>/).

Labels: DFL games from the raw IDSSE npz flags, exactly as bench.dfl_prepared builds them (no bench
cache is read); Metrica dev from tune_on_sample_data.practice_labels. Scoring uses
possession_metrics' own spell_scores / macro_f1 on the abstention-filled prediction, as evaluate().
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENS = os.path.dirname(HERE)
CAND = os.path.dirname(ENS)
ROOT = os.path.dirname(CAND)
for p in (ROOT, os.path.join(ROOT, "evaluation"), CAND, ENS):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.dont_write_bytecode = True
import possession_metrics as pm   # noqa: E402
import members                    # noqa: E402
import ensemble_core as ec        # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

SCRATCH = WORK
COLLECT = os.path.join(SCRATCH, "arch", "ensemble_vote", "collect")
DFL_DIR = os.path.join(SCRATCH, "idsse", "npz")
MATCHES = sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))
DEV_TAGS = ["dev_Sample_Game_1_2027", "dev_Sample_Game_1_2028", "dev_Sample_Game_2_2027",
            "dev_Sample_Game_2_2028"]
FPS = 25

_LDE = []


def lde():
    if not _LDE:
        _LDE.append(members.load("learned_dfl_evidence"))
    return _LDE[0]


def dfl_gt(match):
    z = np.load(os.path.join(DFL_DIR, match + ".npz"))
    status, poss, half = z["status"], z["poss"], z["half"]
    gt = np.full(len(status), pm.X, np.int8)
    gt[(status == 1) & (poss == 1)] = pm.H
    gt[(status == 1) & (poss == 2)] = pm.A
    gt[status == 0] = pm.D
    blocks = []
    for h in (1, 2):
        idx = np.where((half == h) & (gt != pm.X))[0]
        if len(idx):
            blocks.append((len(blocks) + 1, int(idx[0]), int(idx[-1])))
    return gt, blocks


def dev_gt(tag):
    import tune_on_sample_data as tsd
    name = "_".join(tag.split("_")[1:4])
    game = tsd.load_csv_game(os.path.join(SCRATCH, "sample", "sample-data-master", "data", name))
    gt, blocks = tsd.practice_labels(game)
    return np.asarray(gt, np.int8), [tuple(int(v) for v in b) for b in blocks]


def score(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
    s = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, np.asarray(pred, np.int8)))
    return dict(spell=s["mean_f1"], macro=pm.macro_f1(cm3), n_pred=s["n_pred"], n_gt=s["n_gt"])


class Split(object):
    """All members' collected outputs on one split ('dfl' or 'dev')."""

    def __init__(self, split, slugs):
        self.split = split
        self.tags = ["dfl_{}_2027".format(m) for m in MATCHES] if split == "dfl" else DEV_TAGS
        self.gt, self.blocks, self.codes, self.proba, self.side = {}, {}, {}, {}, {}
        dev_cache = {}
        for t in self.tags:
            if split == "dfl":
                self.gt[t], self.blocks[t] = dfl_gt(t[4:-5])
            else:
                name = "_".join(t.split("_")[1:4])
                if name not in dev_cache:
                    dev_cache[name] = dev_gt(t)
                self.gt[t], self.blocks[t] = dev_cache[name]
        self.slugs = []
        for s in slugs:
            files = [os.path.join(COLLECT, s, t + ".npz") for t in self.tags]
            if not all(os.path.isfile(f) for f in files):
                print("[fit] {}: missing {} outputs, skipped".format(s, split))
                continue
            self.slugs.append(s)
            for t, f in zip(self.tags, files):
                z = np.load(f)
                self.codes[(s, t)] = z["codes"].astype(np.int8)
                self.proba[(s, t)] = z["proba"].astype(np.float64)
                if s == "learned_dfl_evidence":
                    self.side[t] = {k: z[k] for k in z.files if k not in ("codes", "proba")}
        self.member_scores = {s: {t: score(self.gt[t], self.codes[(s, t)]) for t in self.tags}
                              for s in self.slugs}

    def hard(self, t, slugs, smooth=None, weights=None):
        return ec.hard_vote([self.codes[(s, t)] for s in slugs], self.blocks[t], lde(), FPS,
                            weights, smooth)

    def soft(self, t, slugs, mode="arith", weights=None, cfg=None):
        P = ec.pool([self.proba[(s, t)] for s in slugs], weights, mode)
        return ec.soft_decode(P, self.side[t], self.blocks[t], len(P), FPS, lde(), cfg)


class Cache(object):
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.isfile(path) else {}
        self.dirty = 0

    def get(self, key, fn):
        if key not in self.d:
            self.d[key] = fn()
            self.dirty += 1
            if self.dirty >= 20:
                self.save()
        return self.d[key]

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"))
        os.replace(tmp, self.path)
        self.dirty = 0
