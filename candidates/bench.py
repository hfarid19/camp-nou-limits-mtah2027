#!/usr/bin/env python3
"""Shared bench for candidate possession pipelines. Every candidate is scored the same way.

A candidate is a file exposing

    predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps) -> codes

    players     (N, slots, 2) float32 metres, NaN where missing (anonymous, fragmented slots)
    slot_team   (slots,) 0 home, 1 away
    ball        (N, 2) float32 metres, NaN where missing (~41% of graded frames)
    sign_half1  +1 if home attacks towards x = pitch length in half 1, else -1
    blocks      [(half, first_frame, last_frame)] inclusive, the graded halves
    tags        Smart Tagging as possession_pipeline.read_tags returns it ([] on sample games)
    names       (home_name, away_name)
    pitch       (length_m, width_m)
    fps         25
    codes       (N,) int  0 H, 1 A, 2 D, 3 X   (possession_metrics convention)

and optionally, for the ensemble,

    predict_proba(same arguments) -> (N, 3) float  P(H), P(A), P(D) per frame

the per-frame evidence the candidate's own decoder works from (posterior marginals for segment
models), rows summing to 1 inside the graded blocks (anything finite outside them). With
--save-probs DIR every scored game's matrix is written to DIR/<split>_<game>_<seed>.npy
(float32) and the match's to DIR/match.npy, and the predicted codes to the same name with
_codes.npy appended (for every candidate, with or without predict_proba).

Usage

    python candidates/bench.py candidates/<name>/pipeline.py            # dev score (Games 1, 2)
    python candidates/bench.py candidates/<name>/pipeline.py --match    # also write the match CSV
    python candidates/bench.py candidates/<name>/pipeline.py --holdout  # Game 3, final check only
    python candidates/bench.py candidates/<name>/pipeline.py --dfl --folds folds.json
                                                                        # 7 open DFL games, LOGO
    ... --save-probs DIR                                                # also save predict_proba
    python candidates/bench.py candidates/<name>/pipeline.py --dfl2 --folds folds.json
                                                                        # DFL, degrade_v2 x 3 seeds

Dev = Metrica open Sample Games 1 and 2, each degraded with two seeds to look like the
competition's automatic tracking (see tune_on_sample_data.degrade). Labels are practice labels
built from those games' event data. Nothing here reads the competition ground truth: the public
5 minutes are a correctness check only (RULES.md), so --match writes the CSV and its format and
possession counts, never a score.

--dfl scores the 7 open DFL/IDSSE Bundesliga games (Bassek et al. 2025, CC BY 4.0), whose labels
come from the tracking feed's own BallStatus / BallPossession flags -- the same meaning as the
competition ground truth. Each game is degraded with the same tune_on_sample_data.degrade as the
dev bench, identically for every candidate. Candidates load LightGBM models trained on these
games, so every Booster a candidate loads must be mapped in --folds (JSON: model sha256 prefix ->
[folds dir, kind]) to the leave-one-game-out fold model that never saw the scored game; an
unmapped model aborts the run instead of scoring on its own training data. The same holds for
every torch.load(path) during --dfl (fold file fold_<kind>_<game index><ext>), and for any
other DFL-trained file, which a candidate must load through candidates/dfl_folds.resolve().

--ctrl T (with --dfl / --dfl2) scores against "controlled possession" labels: every possession
spell shorter than T seconds that follows the other team's possession is folded back into it
(controlled_labels). CHALLENGE.md defines a change of possession as the opponent gaining
CONTROLLED possession (a touch, a deflection or a won aerial duel without control does not
count; proxy: "a player keeps the ball for a few seconds"), whereas the DFL flag flips on every
touch. T is not known, so DFL runs always also report spellF1 against T = 1, 2, 3 s and their
mean, ctrl_mean, which is the selection criterion from 2026-11-08.

--dfl2 is the same, but degrades each game with degrade_v2 (fitted to the competition tracking's
label-free statistics, see degrade_v2.py and match_stats.py) under 3 seeds, and reports the mean
with a standard error over games. It is the primary selection bench from 2026-11-08 on.

--dfl3 is --dfl2 with degrade_v3: the same ball model, but player tracks also break when players
come close together, as the competition tracker's do (62% of its tracklets end in a crowd).
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # dfl_folds
import possession_metrics as pm      # noqa: E402
import possession_pipeline as pp     # noqa: E402
import tune_on_sample_data as tsd    # noqa: E402

# Large never-committed files (DFL data, fold models, caches): $MTAH_WORK, else <repo>/external/work
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SCRATCH = WORK
SAMPLE_DIR = os.environ.get("BENCH_SAMPLE_DIR", os.path.join(SCRATCH, "sample", "sample-data-master", "data"))
CACHE_DIR = os.environ.get("BENCH_CACHE", os.path.join(SCRATCH, "bench_cache"))
DEV = ["Sample_Game_1", "Sample_Game_2"]
HOLDOUT = ["Sample_Game_3"]
SEEDS = [2027, 2028]
DFL_DIR = os.environ.get("BENCH_DFL_DIR", os.path.join(SCRATCH, "idsse", "npz"))
DFL_SEEDS = [2027]
DFL2_SEEDS = [2027, 2028, 2029]


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(module)
    return module


def prepared(name, seed):
    """Degraded sample game, cached: slots, slot_team, ball, gt, blocks, sign."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "{}_{}.npz".format(name, seed))
    if not os.path.isfile(path):
        folder = os.path.join(SAMPLE_DIR, name)
        game = (tsd.load_epts_game(folder) if name == "Sample_Game_3"
                else tsd.load_csv_game(folder))
        gt, blocks = tsd.practice_labels(game)
        tsd.SEED = seed                                  # degrade's ball gaps use the module seed
        slots, slot_team, ball = tsd.degrade(game, gt != pm.X, np.random.default_rng(seed))
        sign = tsd.attack_sign(game, blocks)
        np.savez(path, slots=slots, slot_team=slot_team, ball=ball, gt=gt,
                 blocks=np.array(blocks), sign=sign)
    z = np.load(path)
    return (z["slots"], z["slot_team"], z["ball"], z["gt"],
            [tuple(int(v) for v in b) for b in z["blocks"]], float(z["sign"]))


def dfl_prepared(match, seed):
    """Degraded DFL game, cached; labels D where BallStatus 0, else H/A from BallPossession."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "dfl_{}_{}.npz".format(match, seed))
    if not os.path.isfile(path):
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
        game = tsd.Game(match, z["players"].astype(np.float64), z["team_of_player"].astype(int),
                        z["ball"].astype(np.float64), [])
        tsd.SEED = seed
        slots, slot_team, ball = tsd.degrade(game, gt != pm.X, np.random.default_rng(seed))
        sign = tsd.attack_sign(game, blocks)
        np.savez(path, slots=slots, slot_team=slot_team, ball=ball, gt=gt,
                 blocks=np.array(blocks), sign=sign)
    z = np.load(path)
    return (z["slots"], z["slot_team"], z["ball"], z["gt"],
            [tuple(int(v) for v in b) for b in z["blocks"]], float(z["sign"]))


def save_probs(module, probs_dir, tag, args, codes=None):
    """Write predict_proba's (N, 3) matrix for one scored game, if the candidate has one, and the
    predicted codes as <tag>_codes.npy (int8)."""
    if probs_dir and codes is not None:
        os.makedirs(probs_dir, exist_ok=True)
        np.save(os.path.join(probs_dir, tag + "_codes.npy"), np.asarray(codes, np.int8))
    if not probs_dir or not hasattr(module, "predict_proba"):
        return
    p = np.asarray(module.predict_proba(*args), np.float32)
    assert p.shape == (len(args[2]), 3), p.shape
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, tag + ".npy"), p)


def dfl2_prepared(match, seed, version=2):
    """DFL game degraded with degrade_v2 (ball hidden around dead-ball runs), cached. version=3:
    degrade_v3 (same ball, occlusion-driven player fragmentation)."""
    import degrade_v2
    import degrade_v3
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "dfl{}_{}_{}.npz".format(version, match, seed))
    if not os.path.isfile(path):
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
        players, ball = z["players"].astype(np.float64), z["ball"].astype(np.float64)
        team = z["team_of_player"].astype(int)
        fn = degrade_v3.degrade_v3 if version == 3 else degrade_v2.degrade_v2
        slots, slot_team, dball = fn(players, team, ball, status, gt != pm.X,
                                                         np.random.default_rng(seed))
        sign = tsd.attack_sign(tsd.Game(match, players, team, ball, []), blocks)
        np.savez(path, slots=slots, slot_team=slot_team, ball=dball, gt=gt,
                 blocks=np.array(blocks), sign=sign)
    z = np.load(path)
    return (z["slots"], z["slot_team"], z["ball"], z["gt"],
            [tuple(int(v) for v in b) for b in z["blocks"]], float(z["sign"]))


def controlled_labels(gt, T, fps=25):
    """Ground truth under the challenge's control rule (control.fold on its own graded runs)."""
    import control
    return control.fold(gt, T, pm.valid_runs(gt), fps)


class _FoldBooster(object):
    """A leave-one-game-out fold model standing in for a full model, cut to the full model's rounds."""

    def __init__(self, fold, rounds):
        self.fold, self.rounds = fold, rounds

    def predict(self, X, num_iteration=None, **kw):
        it = self.rounds if not num_iteration or num_iteration <= 0 else min(num_iteration, self.rounds)
        return self.fold.predict(X, num_iteration=it, **kw)

    def __getattr__(self, name):
        return getattr(self.fold, name)


def bench_dfl(path, folds, folds_path=None, probs_dir=None, v2=False, ctrl=0.0):
    import hashlib
    import lightgbm as lgb
    import dfl_folds
    real = lgb.Booster
    torch_real = {}    # torch is never imported here: its libomp clashes with LightGBM's on macOS

    def torch_load(f, *args, **kw):
        if not isinstance(f, (str, os.PathLike)):
            raise RuntimeError("bench --dfl only supports torch.load(path)")
        return torch_real["load"](dfl_folds.resolve(os.fspath(f)), *args, **kw)

    def guard_torch():
        torch = sys.modules.get("torch")
        if torch is None:
            return
        if "load" not in torch_real:
            torch_real["load"], torch_real["module"] = torch.load, torch
        if torch.load is not torch_load:
            if "patched" in torch_real:
                raise RuntimeError("torch.load was replaced during a DFL game")
            torch.load = torch_load
        torch_real["patched"] = True
    matches = sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))
    cand_dir = os.path.dirname(os.path.abspath(path))
    rows = []
    only = [m for m in os.environ.get("BENCH_DFL_ONLY", "").split(",") if m]   # opt-in: score a subset (fold index stays global)
    for gi, match in enumerate(matches):
        if only and match not in only:
            continue
        def booster(*args, **kw):
            model_file = kw.get("model_file")
            if args or model_file is None:
                raise RuntimeError("bench --dfl only supports Booster(model_file=...)")
            with open(model_file, "rb") as handle:
                sha = hashlib.sha256(handle.read()).hexdigest()
            key = next((k for k in folds if sha.startswith(k)), None)
            if key is None:
                raise RuntimeError("{} (sha256 {}) has no fold mapping in --folds: refusing to "
                                   "score DFL games it was trained on".format(model_file, sha[:12]))
            folder, kind = folds[key]
            full = real(model_file=model_file)
            fold = real(model_file=os.path.join(folder, "fold_{}_{}.txt".format(kind, gi)))
            if fold.feature_name() != full.feature_name():
                raise RuntimeError("fold {} {} features differ from {}".format(kind, gi, model_file))
            return _FoldBooster(fold, full.num_trees())
        for name in [m for m, mod in sys.modules.items()     # fresh module state per game
                     if getattr(mod, "__file__", None) and
                     os.path.abspath(mod.__file__).startswith(cand_dir + os.sep)]:
            del sys.modules[name]
        lgb.Booster = booster
        os.environ["BENCH_DFL_FOLD"] = str(gi)
        os.environ["BENCH_DFL_FOLDS"] = os.path.abspath(folds_path) if folds_path else os.devnull
        torch_real.pop("patched", None)
        guard_torch()
        try:
            module = load_candidate(path)
            guard_torch()
            for seed in (DFL2_SEEDS if v2 else DFL_SEEDS):
                slots, slot_team, ball, gt, blocks, sign = (dfl2_prepared(match, seed, v2) if v2 else dfl_prepared(match, seed))
                args = (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
                t0 = time.time()
                pred = np.asarray(module.predict(*args), np.int8)
                secs = time.time() - t0
                save_probs(module, probs_dir, "{}_{}_{}".format("dfl{}".format(v2) if v2 else "dfl", match, seed), args,
                           pred)
                if "torch" in sys.modules and "patched" not in torch_real:
                    raise RuntimeError("import torch at the top of the candidate (or not at all) "
                                       "so bench --dfl can map torch.load to the fold models")
                ctrls = [pm.evaluate(controlled_labels(gt, T), pred, bootstrap_n=0)["spell_f1_mean"]
                         for T in (1, 2, 3)]
                gt = controlled_labels(gt, ctrl)
                r = pm.evaluate(gt, pred, bootstrap_n=0)
                rows.append(dict(game=match, seed=seed, spellF1=round(r["spell_f1_mean"], 4),
                                 macroF1=round(r["macro_f1"], 4),
                                 ctrl=[round(v, 4) for v in ctrls], ctrl_mean=round(float(np.mean(ctrls)), 4),
                                 poss="{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]),
                                 secs=round(secs, 1)))
                print("  {:14s} seed {}  spellF1 {:.4f}  ctrl {:.4f}  macroF1 {:.4f}  poss pred/gt {}  {:.1f}s".format(
                    match, seed, r["spell_f1_mean"], rows[-1]["ctrl_mean"], r["macro_f1"], rows[-1]["poss"], secs))
        finally:
            lgb.Booster = real
            os.environ.pop("BENCH_DFL_FOLD", None)
            os.environ.pop("BENCH_DFL_FOLDS", None)
            if "load" in torch_real:
                torch_real["module"].load = torch_real["load"]
    if dfl_folds.RESOLVED:
        print("  fold files resolved: " + ", ".join(sorted({"{} -> {}".format(a, os.path.basename(b))
                                                            for a, b in dfl_folds.RESOLVED})))
    return rows


def bench(module, games, split="dev", probs_dir=None):
    rows = []
    for name in games:
        for seed in SEEDS:
            slots, slot_team, ball, gt, blocks, sign = prepared(name, seed)
            args = (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
            t0 = time.time()
            pred = np.asarray(module.predict(*args), np.int8)
            secs = time.time() - t0
            save_probs(module, probs_dir, "{}_{}_{}".format(split, name, seed), args, pred)
            r = pm.evaluate(gt, pred, bootstrap_n=0)
            rows.append(dict(game=name, seed=seed, spellF1=round(r["spell_f1_mean"], 4),
                             macroF1=round(r["macro_f1"], 4),
                             poss="{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]),
                             secs=round(secs, 1)))
            print("  {:14s} seed {}  spellF1 {:.4f}  macroF1 {:.4f}  poss pred/gt {}  {:.1f}s".format(
                name, seed, r["spell_f1_mean"], r["macro_f1"], rows[-1]["poss"], secs))
    return rows


def run_match(module, out_path, probs_dir=None):
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n_frames = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    args = (players, slot_team, ball, sign_half1, blocks, tags, names,
            GAMES[GAME]["pitch"], GAMES[GAME]["fps"])
    codes = np.asarray(module.predict(*args), np.int8)
    save_probs(module, probs_dir, "match", args, codes)
    assert codes.shape == (n_frames,), codes.shape
    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    letters = np.array(pm.LABELS)[codes]
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))
    graded = np.zeros(n_frames, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    runs = [(first, last + 1) for _, first, last in blocks]
    stats = pm.possession_stats(codes, runs)
    summary = dict(rows=n_frames, possessions=stats["n_spells"],
                   mean_poss_s=round(stats["mean_duration_s"], 1),
                   dead_share=round(float((codes[graded] == pm.D).mean()), 3),
                   x_in_halves=int((codes[graded] == pm.X).sum()))
    print("match CSV -> {}  {}".format(out_path, summary))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate")
    ap.add_argument("--holdout", action="store_true", help="score Game 3 (final check only)")
    ap.add_argument("--match", action="store_true", help="also write <candidate dir>/submission.csv")
    ap.add_argument("--dfl", action="store_true", help="score the 7 open DFL games (LOGO models)")
    ap.add_argument("--dfl2", action="store_true", help="as --dfl, degrade_v2 x 3 seeds (primary)")
    ap.add_argument("--dfl3", action="store_true", help="as --dfl2, degrade_v3 (crowd fragmentation)")
    ap.add_argument("--ctrl", type=float, default=0.0,
                    help="score against controlled-possession labels with this T in seconds")
    ap.add_argument("--folds", default=None, help="JSON file: sha256 prefix -> [folds dir, kind]")
    ap.add_argument("--save-probs", default=None, metavar="DIR",
                    help="write each scored game's predict_proba matrix to DIR")
    args = ap.parse_args()
    ver = 3 if args.dfl3 else 2 if args.dfl2 else 0
    if args.dfl or ver:
        folds = json.load(open(args.folds)) if args.folds else {}
        print("{} on DFL{} x seeds {}".format(args.candidate, " (degrade_v{})".format(ver) if ver else "",
                                              DFL2_SEEDS if ver else DFL_SEEDS))
        rows = bench_dfl(args.candidate, folds, args.folds, args.save_probs, v2=ver, ctrl=args.ctrl)
        split = ("dfl{}".format(ver) if ver else "dfl") + ("_ctrl{:g}".format(args.ctrl) if args.ctrl else "")
    else:
        module = load_candidate(args.candidate)
        games = HOLDOUT if args.holdout else DEV
        print("{} on {} x seeds {}".format(args.candidate, games, SEEDS))
        split = "holdout" if args.holdout else "dev"
        rows = bench(module, games, split, args.save_probs)
    result = dict(candidate=args.candidate, split=split,
                  spellF1_mean=round(float(np.mean([r["spellF1"] for r in rows])), 4),
                  macroF1_mean=round(float(np.mean([r["macroF1"] for r in rows])), 4), rows=rows)
    games = sorted({r["game"] for r in rows})
    if len(games) > 1:                                 # standard error over games (seeds averaged)
        per_game = [np.mean([r["spellF1"] for r in rows if r["game"] == g]) for g in games]
        result["spellF1_se"] = round(float(np.std(per_game, ddof=1) / np.sqrt(len(games))), 4)
    if rows and "ctrl_mean" in rows[0]:
        result["ctrl_mean"] = round(float(np.mean([r["ctrl_mean"] for r in rows])), 4)
        result["ctrl_by_T"] = [round(float(np.mean([r["ctrl"][i] for r in rows])), 4) for i in range(3)]
        print("ctrl_mean {} (T=1,2,3: {})".format(result["ctrl_mean"], result["ctrl_by_T"]))
    if args.match and not (args.dfl or ver):
        module = load_candidate(args.candidate)
        out = os.path.join(os.path.dirname(os.path.abspath(args.candidate)), "submission.csv")
        result["match"] = run_match(module, out, args.save_probs)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
