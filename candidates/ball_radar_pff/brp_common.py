"""ball_radar_pff: training games and fold specs (same specs as candidates/nn_pff/data.py).

games()        -> {name: npz path}: the 7 DFL games (DFL-MAT-*) + the 48 PFF selection games
                  (PFF-<id>); the 16 PFF holdout games are never included (asserted).
                  Round 4 opt-in (NN_PFF_ALLOW_HOLDOUT=1, as candidates/nn_pff/data.py): the 16
                  spent holdout games are added (64 PFF games) and SPECS is dfl0-6 only (no PFF folds).
Work folder: $BRP_ARCH (default $MTAH_WORK/arch/ball_radar_pff), so a 64-game retrain lives apart.
excluded(spec) -> names a model of that spec must not see: pff<k> = PFF selection fold k,
                  dfl<g> = DFL game g (index in the sorted DFL list, as bench.py), full = none.
"""
import json
import os

WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")
ARCH = os.path.join(WORK, "arch", "nn_pff")
ALLOW_HOLDOUT = os.environ.get("NN_PFF_ALLOW_HOLDOUT", "") == "1"
SPECS = ([] if ALLOW_HOLDOUT else ["pff0", "pff1", "pff2", "pff3"]) + ["dfl{}".format(g) for g in range(7)]
BRP_ARCH = os.environ.get("BRP_ARCH") or os.path.join(WORK, "arch", "ball_radar_pff")


def holdout():
    return sorted(json.load(open(os.path.join(ARCH, "holdout_games.json")))["games"])


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
    if ALLOW_HOLDOUT:
        for g in hold:
            out["PFF-" + g] = os.path.join(WORK, "pff_wc2022", "npz", "PFF-" + g + ".npz")
    return dict(sorted(out.items()))


def excluded(spec):
    if spec == "full":
        return set()
    if spec.startswith("pff"):
        assert not ALLOW_HOLDOUT, "NN_PFF_ALLOW_HOLDOUT=1: no PFF fold specs"
        return {"PFF-" + g for g in pff_folds()[int(spec[3:])]}
    return {dfl_matches()[int(spec[3:])]}


def train_names(spec):
    return [n for n in games() if n not in excluded(spec)]
