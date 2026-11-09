"""ball_radar_pff: training games and fold specs (same specs as candidates/nn_pff/data.py).

games()        -> {name: npz path}: the 7 DFL games (DFL-MAT-*) + the 48 PFF selection games
                  (PFF-<id>); the 16 PFF holdout games are never included (asserted).
excluded(spec) -> names a model of that spec must not see: pff<k> = PFF selection fold k,
                  dfl<g> = DFL game g (index in the sorted DFL list, as bench.py), full = none.
"""
import json
import os

WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")
ARCH = os.path.join(WORK, "arch", "nn_pff")
SPECS = ["pff0", "pff1", "pff2", "pff3"] + ["dfl{}".format(g) for g in range(7)]


def dfl_matches():
    return sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "idsse", "npz")) if f.endswith(".npz"))


def pff_folds():
    return json.load(open(os.path.join(ARCH, "selection_folds.json")))["folds"]


def games():
    hold = set(json.load(open(os.path.join(ARCH, "holdout_games.json")))["games"])
    out = {m: os.path.join(WORK, "idsse", "npz", m + ".npz") for m in dfl_matches()}
    for f in pff_folds():
        for g in f:
            assert g not in hold
            out["PFF-" + g] = os.path.join(WORK, "pff_wc2022", "npz", "PFF-" + g + ".npz")
    return dict(sorted(out.items()))


def excluded(spec):
    if spec == "full":
        return set()
    if spec.startswith("pff"):
        return {"PFF-" + g for g in pff_folds()[int(spec[3:])]}
    return {dfl_matches()[int(spec[3:])]}


def train_names(spec):
    return [n for n in games() if n not in excluded(spec)]
