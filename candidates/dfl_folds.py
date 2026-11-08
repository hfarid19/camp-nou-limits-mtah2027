"""Leave-one-game-out model lookup for DFL-trained artefacts that are not LightGBM / torch files.

bench.py --dfl already swaps lightgbm.Booster(model_file=...) and torch.load(path) for the fold
model that never saw the scored game. Any other DFL-trained file a candidate loads (numpy
weights, JSON tables, pickles) must go through resolve():

    import dfl_folds
    weights = np.load(dfl_folds.resolve(os.path.join(HERE, "weights.npz")))

Outside the DFL bench resolve() returns the path unchanged. Inside it, the file's sha256 prefix
must be mapped in the --folds JSON to [folds dir, kind], and the fold file
<folds dir>/fold_<kind>_<game index><ext> is returned; an unmapped file raises.
"""

import hashlib
import json
import os

RESOLVED = []


def lookup(path):
    """Fold file for path while bench --dfl scores a game, else None."""
    gi = os.environ.get("BENCH_DFL_FOLD")
    if gi is None:
        return None
    with open(os.environ["BENCH_DFL_FOLDS"]) as handle:
        text = handle.read()
    folds = json.loads(text) if text.strip() else {}
    with open(path, "rb") as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    key = next((k for k in folds if sha.startswith(k)), None)
    if key is None:
        raise RuntimeError("{} (sha256 {}) has no fold mapping in --folds: refusing to score DFL "
                           "games it was trained on".format(path, sha[:12]))
    folder, kind = folds[key]
    fold = os.path.join(folder, "fold_{}_{}{}".format(kind, gi, os.path.splitext(path)[1]))
    RESOLVED.append((os.path.basename(path), fold))
    return fold


def resolve(path):
    fold = lookup(path)
    return path if fold is None else fold
