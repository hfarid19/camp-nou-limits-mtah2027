#!/usr/bin/env python3
"""PFF runner for nn_asformer (= candidates/nn_pff/run_pff.py with the output root changed to
$MTAH_WORK/probs/nn_asformer; the fold protocol, env variables and checks are identical).
Out-of-fold on the 48 selection games.

    python candidates/nn_asformer/run_pff.py <pipeline.py> --name NAME --folds <pff_folds.json> \
        [--model <full.npz>] [--games G ...]
    python candidates/nn_asformer/run_pff.py <pipeline.py> --name NAME --holdout --final   # once, at the end

Selection mode (default): for each selection game (arch/nn_pff/selection_folds.json) the env tells
pff_folds.resolve to load fold_pff_<k>.npz, k = the game's fold, i.e. a model trained without the
game's 12-game fold; --folds maps a weight file's sha256 prefix to the folder of its fold models.
After every game the runner checks NN_PFF_LOG: the game must have been resolved to an
out-of-fold model (otherwise it stops), and pff_folds itself refuses a model whose train_games
contain the game. Other members (DFL-trained only) never saw PFF and use their full models;
BENCH_DFL_FOLD must not be set. --model overrides a single-member pipeline's MODEL_PATH (bench
wrappers of the variants, as temporal_tcn_v2/pipeline_raw.py does).

Holdout mode (--holdout --final): the 16 holdout games with the full (shipped) models and no
fold lookup; refuses to run without --final, and refuses if NAME already has holdout outputs.

Inputs = candidates/pff/bench_pff.pff2_prepared (degrade_v2, seed 2027, the PFF bench cache).
Outputs: $MTAH_WORK/probs/nn_asformer/NAME/pff2_<game>_2027_codes.npy and .npy (predict_proba).
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND, ROOT, os.path.join(ROOT, "evaluation")]
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
ARCH = os.path.join(WORK, "arch", "nn_pff")
PROBS = os.path.join(WORK, "probs", "nn_asformer")
SEED = 2027

import possession_metrics as pm   # noqa: E402
import bench_pff                  # noqa: E402


def fold_of():
    return {g: k for k, f in enumerate(json.load(open(os.path.join(ARCH, "selection_folds.json")))["folds"]) for g in f}


def load_candidate(path, name):
    spec = importlib.util.spec_from_file_location("nnpff_cand_" + name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-proba", action="store_true")
    a = ap.parse_args()
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS", "NN_PFF_FOLD"):
        if k in os.environ:
            raise SystemExit(k + " is set")
    import tune_on_sample_data as tsd
    hold = json.load(open(os.path.join(ARCH, "holdout_games.json")))["games"]
    out = os.path.join(PROBS, a.name)
    os.makedirs(out, exist_ok=True)
    if a.holdout:
        if not a.final:
            raise SystemExit("--holdout needs --final: the holdout is scored once, for the final pick")
        games = hold
        if any(os.path.exists(os.path.join(out, "pff2_{}_{}_codes.npy".format(g, SEED))) for g in games):
            raise SystemExit("holdout outputs already exist for " + a.name)
        fmap = None
    else:
        fmap = fold_of()
        games = a.games or sorted(fmap)
        assert not set(games) & set(hold), "holdout games requested in selection mode"
        if not a.folds:
            raise SystemExit("--folds is required in selection mode")
        os.environ["NN_PFF_FOLDS"] = os.path.abspath(a.folds)
        os.environ["NN_PFF_LOG"] = os.path.join(out, "resolve.log")
    todo = [g for g in games if not os.path.isfile(os.path.join(out, "pff2_{}_{}_codes.npy".format(g, SEED)))]
    print("{} -> {}: {} games to run".format(a.pipeline, out, len(todo)), flush=True)
    if not todo:
        return
    module = load_candidate(a.pipeline, a.name)
    if a.model:                      # MS-TCN pipeline: MODEL_PATH; TCN pipeline: WEIGHTS
        attr = "MODEL_PATH" if hasattr(module, "MODEL_PATH") else "WEIGHTS"
        setattr(module, attr, os.path.abspath(a.model))
    for g in todo:
        slots, slot_team, ball, gt, blocks, sign = bench_pff.pff2_prepared(g)
        args = (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
        if fmap is not None:
            os.environ["NN_PFF_FOLD"] = str(fmap[g])
            os.environ["NN_PFF_GAME"] = g
            n0 = sum(1 for _ in open(os.environ["NN_PFF_LOG"])) if os.path.exists(os.environ["NN_PFF_LOG"]) else 0
        t0 = time.time()
        pred = np.asarray(module.predict(*args), np.int8)
        P = None if a.no_proba or not hasattr(module, "predict_proba") else np.asarray(module.predict_proba(*args), np.float32)
        secs = time.time() - t0
        if fmap is not None:
            lines = open(os.environ["NN_PFF_LOG"]).read().splitlines()[n0:]
            if not any(l.split()[0] == g for l in lines):
                raise SystemExit("game {}: no out-of-fold model was resolved (cached full model?)".format(g))
            os.environ.pop("NN_PFF_FOLD")
        np.save(os.path.join(out, "pff2_{}_{}_codes.npy".format(g, SEED)), pred)
        if P is not None:
            np.save(os.path.join(out, "pff2_{}_{}.npy".format(g, SEED)), P)
        if not a.no_proba and hasattr(module, "boundary_proba"):
            np.save(os.path.join(out, "pff2_{}_{}_pb.npy".format(g, SEED)),
                    np.asarray(module.boundary_proba(*args), np.float32))
        r = pm.evaluate(gt, pred, bootstrap_n=0)
        print("  {:6s} fold {} spellF1 {:.4f} macroF1 {:.4f} poss {}/{} {:.1f}s".format(
            g, fmap[g] if fmap else "H", r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], secs),
            flush=True)


if __name__ == "__main__":
    main()
