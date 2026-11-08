"""Turn fit.py's nested results into the shipped rule files.

    python write_configs.py <kind> <label index in nested_<kind>.json>

config_<kind>.json (repo)   the rule selected on all 7 DFL games (used on dev / holdout / match)
fold_cfg<kind>_<gi>.json    the rule selected on the 6 games other than gi (scratch), mapped in
                            arch/ensemble_vote/folds.json by config_<kind>.json's sha256 so that
                            bench --dfl scores game gi with the rule that never saw it
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitlib as F   # noqa: E402
import fit           # noqa: E402

FOLD_DIR = os.path.join(F.SCRATCH, "arch", "ensemble_vote", "folds_cfg")
FOLDS_JSON = os.path.join(F.SCRATCH, "arch", "ensemble_vote", "folds.json")


def rule(kind, slugs, opt, train, sp, note):
    cfg = dict(kind=kind, slugs=list(slugs), note=note)
    if kind == "hard":
        cfg["weights"] = None
        cfg["smooth"] = fit.SMOOTH[opt["smooth"]]
    else:
        cfg["weights"] = fit.weights_for(sp, slugs, train, opt["w"])
        cfg["pool"] = opt["pool"]
        cfg["decoder"] = fit.DEC[opt["dec"]]
    cfg["opt"] = opt
    return cfg


def main():
    kind, idx = sys.argv[1], int(sys.argv[2])
    res = json.load(open(os.path.join(fit.OUT, "nested_{}.json".format(kind))))[idx]
    sp = F.Split("dfl", F.members.ALL)
    os.makedirs(FOLD_DIR, exist_ok=True)
    for gi, r in enumerate(res["rows"]):
        assert r["game"] == sp.tags[gi]
        train = [t for t in sp.tags if t != r["game"]]
        cfg = rule(kind, r["slugs"], r["opt"], train, sp,
                   "nested fold {}: selected on the 6 DFL games other than {} ({}, train {:.4f})".format(
                       gi, r["game"], r["how"], r["train"]))
        json.dump(cfg, open(os.path.join(FOLD_DIR, "fold_cfg{}_{}.json".format(kind, gi)), "w"),
                  indent=1)
    fin = res["final"]
    cfg = rule(kind, fin["slugs"], fin["opt"], list(sp.tags), sp,
               "selected on all 7 DFL games ({}, in-sample {:.4f}; nested LOGO estimate {:.4f})".format(
                   fin["how"], fin["in_sample"], res["nested"]))
    path = os.path.join(F.ENS, "config_{}.json".format(kind))
    with open(path, "w") as handle:
        json.dump(cfg, handle, indent=1)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    folds = json.load(open(FOLDS_JSON))
    folds = {k: v for k, v in folds.items() if v[1] != "cfg" + kind}
    folds[sha] = [FOLD_DIR, "cfg" + kind]
    json.dump(folds, open(FOLDS_JSON, "w"), indent=1, sort_keys=True)
    print(path, sha, json.dumps(cfg))


if __name__ == "__main__":
    main()
