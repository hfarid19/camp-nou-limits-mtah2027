"""Diagnostic (not a selection target): v2's full predict() on leave-one-DFL-game-out evidence.

For each of the 7 open DFL/IDSSE games: degrade it with seed 21 (never used for training) using
learned_dfl_evidence/make_dataset.degrade_dfl, swap v2's frozen boosters for the fold models
trained WITHOUT that game (<cache>/folds/fold_{d,ha,c}_<i>.txt, truncated to the rounds in
lde_frozen/model_meta.json), run predict and score against the game's own tracking flags
(BallStatus / BallPossession: the same label semantics as the real ground truth).

    [MD_DIR=<dir with make_dataset.py>] [DFL_DEGRADE='{"BLACKOUT_P": 0, "DROP_P": 0}'] \
    python -I candidates/v2_stacked_fixes/tuning/dfl_logo.py <idsse cache with npz/ and folds/> '{cfg}' ...
Round 2 used a snapshot of learned_dfl_evidence/make_dataset.py (md5 fc2c3f39...) and the
round-1 fold models (the ones matching lde_frozen/), on two degradations: "r1" (DROP_P 0,
BLACKOUT_P 0: the round-1 training distribution) and "r2" (DROP_P 0.5, BLACKOUT_P 0.5: extra
dropouts and whole-stoppage ball blackouts, closer to the match's long gaps).
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(V2))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
# MD_DIR: directory of the make_dataset.py snapshot to degrade with (default: the live candidate)
sys.path.insert(0, os.environ.get("MD_DIR", os.path.join(ROOT, "candidates", "learned_dfl_evidence")))
sys.path.insert(0, V2)
os.environ.setdefault("V2_QUIET", "1")
warnings.filterwarnings("ignore")
import possession_metrics as pm  # noqa: E402
import make_dataset as md  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("v2_pipeline", os.path.join(V2, "pipeline.py"))
pipeline = importlib.util.module_from_spec(_spec)   # v2's pipeline, not learned_dfl_evidence's
_spec.loader.exec_module(pipeline)


class _Trunc(object):
    def __init__(self, booster, rounds):
        self.b, self.k = booster, rounds

    def predict(self, X, num_threads=1):
        return self.b.predict(X, num_iteration=self.k, num_threads=num_threads)


def main():
    import lightgbm as lgb
    for k, v in json.loads(os.environ.get("DFL_DEGRADE", "{}")).items():   # e.g. {"BLACKOUT_P": 0}
        assert hasattr(md, k), k
        setattr(md, k, v)
    cache = sys.argv[1]
    cfgs = [json.loads(a) for a in sys.argv[2:]] or [{}]
    meta = json.load(open(os.path.join(V2, "lde_frozen", "model_meta.json")))
    rows = {i: [] for i in range(len(cfgs))}
    for gi, match in enumerate(meta["matches"]):
        g = md.load_game(os.path.join(cache, "npz", match + ".npz"))
        slots, slot_team, ball, codes, blocks, sign, _ = md.degrade_dfl(g, 21, md.game_key(match))
        pipeline._lde_logits(slots[:500], slot_team, ball[:500], sign, [(1, 0, 499)], md.PITCH, md.FPS, 5.0)
        for w in ("d", "ha", "c"):
            b = lgb.Booster(model_file=os.path.join(cache, "folds", "fold_{}_{}.txt".format(w, gi)))
            pipeline._LDE[w] = _Trunc(b, meta[w]["rounds"])
        for i, cfg in enumerate(cfgs):
            pred = pipeline.predict(slots, slot_team, ball, sign, blocks, [], ("Home", "Away"),
                                    md.PITCH, md.FPS, cfg=cfg)
            r = pm.evaluate(codes, np.asarray(pred, np.int8), bootstrap_n=0)
            rows[i].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
            print("{} cfg {}: spellF1 {:.4f} macroF1 {:.4f} poss {}/{}".format(match, i, *rows[i][-1]),
                  flush=True)
    for i, cfg in enumerate(cfgs):
        a = np.array(rows[i])
        print("DFL-LOGO {}: spellF1 {:.4f} macroF1 {:.4f} poss {:.0f}/{:.0f}".format(
            json.dumps(cfg), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].sum(), a[:, 3].sum()))


if __name__ == "__main__":
    main()
