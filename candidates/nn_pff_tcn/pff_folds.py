"""Model lookup for PFF+DFL-trained weights: DFL leave-one-game-out (bench.py) and PFF out-of-fold
(candidates/nn_pff/run_pff.py).

resolve(path, kind=None, dfl=True):
- dfl=True first applies candidates/dfl_folds.resolve (bench --dfl / --dfl2 sets BENCH_DFL_FOLD: the
  fold model that never saw the scored DFL game). LightGBM files pass dfl=False: under the DFL bench
  bench.py itself swaps lightgbm.Booster for the fold model.
- Under run_pff.py (NN_PFF_FOLD = k set) it returns <folder>/fold_pff_<k><ext> (kind None) or
  <folder>/fold_pff_<kind>_<k><ext>, for the folder mapped to the file's sha256 prefix in
  NN_PFF_FOLDS, after checking that the fold model's training games (the npz key `train_games`, or
  the sidecar <fold file>.games.json) do not contain the scored game NN_PFF_GAME. Every lookup is
  appended to NN_PFF_LOG ("<game> <fold file>"), which the runner checks after each game.
- Otherwise the path is returned unchanged (the match, Metrica, the PFF holdout: full models).
The sha256 is that of the file's content, or of its re-joined parts when it is stored split
(candidates/chunked_weights.py), so the lookup is the same for both forms.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chunked_weights  # noqa: E402  (a weights file may be stored split into parts; see chunked_weights.py)
import dfl_folds        # noqa: E402


def train_games(fold):
    side = fold + ".games.json"
    if os.path.isfile(side):
        return set(json.load(open(side)))
    return {str(g) for g in chunked_weights.open_weights(fold)["train_games"]}


def resolve(path, kind=None, dfl=True):
    out = dfl_folds.resolve(path) if dfl else path
    k = os.environ.get("NN_PFF_FOLD")
    if k is None:
        return out
    if out != path or os.environ.get("BENCH_DFL_FOLD") is not None:
        raise RuntimeError("both the DFL and the PFF fold lookups are active")
    sha = chunked_weights.sha256(path)
    folds = json.load(open(os.environ["NN_PFF_FOLDS"]))
    key = next((s for s in folds if sha.startswith(s)), None)
    if key is None:
        raise RuntimeError("{} (sha256 {}) has no PFF fold mapping".format(path, sha[:12]))
    stem = "fold_pff_{}".format(int(k)) if kind is None else "fold_pff_{}_{}".format(kind, int(k))
    fold = os.path.join(folds[key], stem + os.path.splitext(path)[1])
    game = os.environ["NN_PFF_GAME"]
    if "PFF-" + game in train_games(fold):
        raise RuntimeError("{} was trained on PFF-{}".format(fold, game))
    with open(os.environ["NN_PFF_LOG"], "a") as h:
        h.write("{} {}\n".format(game, fold))
    return fold
