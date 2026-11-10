#!/usr/bin/env python3
"""Write Camp Nou Limits' submission CSV: candidates/final_f4b ("F4b").

    pip install -r requirements-submission.txt
    python make_submission.py                    # writes evaluation/results/camp_nou_limits.csv
    python evaluation/validate_submission.py evaluation/results/camp_nou_limits.csv

Reads only the competition input (ATD meta + raw tracking, Smart Tagging) through games.py and
possession_pipeline's readers, runs candidates/final_f4b/pipeline.predict once and writes
frame,label rows. F4b runs four members, each trained on the 7 DFL + 48 PFF World Cup 2022 games:
  - the label-smoothed TCN (nn_pff_tcn, tcn_ls.npz), averaged over the input and 4 jittered copies
    of it, with its feature normalisation moved half-way towards the match's own label-free
    feature statistics (candidates/adapt/member_tcn_tta4_a50.py);
  - the MS-TCN at width 192 with label smoothing (nn_pff, mstcn_c192ls.npz);
  - the ball radar with stacked LightGBM (ball_radar_pff);
  - the ball-centred possessor-pointer network (nn_pointer, ptr_ballc.npz).
It then decodes them with candidates/mbr_decode: per stretch of play it keeps the candidate label
sequence (the members' hard vote under v6's rule of 1 s smoothing + 1.25 s controlled-possession
fold, the vote under 11 other smoothing/fold rules, or a single member) with the highest expected
spellF1 against samples from an HMM on the members' mean probabilities (fixed seed). Last, every
frame before each half's kick-off is dead. No ground truth, no training data and no network access
are needed: the members load their committed model files (numpy + LightGBM only).

Numerical libraries are pinned to one thread before numpy is imported. That alone does not make
floating-point results machine-independent; what we tested is that the file is byte-identical
(sha256 228fe6a47e29a411746a97e9c7f758ad557c2f6acf768343b39932ede8ac4f30) on macOS arm64
(Accelerate), Linux arm64 (OpenBLAS) and Linux x86-64 under Rosetta emulation (OpenBLAS), with the
pinned requirements-submission.txt and Python 3.14.6. This script always runs single-threaded; the
2- and 8-thread runs that gave the same file on macOS went through candidates/final_f4b/match_run.py
(the same pipeline without the thread pin). Native x86-64 hardware was not tested. See
TECHNICAL.md, "Reproduction".
"""

import os

for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse          # noqa: E402
import importlib.util    # noqa: E402
import sys               # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.dont_write_bytecode = True
import possession_metrics as pm      # noqa: E402
import possession_pipeline as pp     # noqa: E402


def load_ensemble():
    """Load candidates/final_f4b/pipeline.py under its own name, as candidates/bench.py does:
    the vote's member loader clears every module called "pipeline" while it loads the members."""
    path = os.path.join(ROOT, "candidates", "final_f4b", "pipeline.py")
    spec = importlib.util.spec_from_file_location("camp_nou_limits_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "camp_nou_limits.csv"))
    args = ap.parse_args()
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n_frames = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    ensemble = load_ensemble()
    codes = np.asarray(ensemble.predict(players, slot_team, ball, sign_half1, blocks, tags, names,
                                        GAMES[GAME]["pitch"], GAMES[GAME]["fps"]), np.int8)
    assert codes.shape == (n_frames,), codes.shape
    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    letters = np.array(pm.LABELS)[codes]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))
    graded = np.zeros(n_frames, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    stats = pm.possession_stats(codes, [(first, last + 1) for _, first, last in blocks])
    print("submission -> {}  possessions {}, mean {:.1f} s, dead share {:.3f}, X in halves {}".format(
        args.out, stats["n_spells"], stats["mean_duration_s"],
        float((codes[graded] == pm.D).mean()), int((codes[graded] == pm.X).sum())))


if __name__ == "__main__":
    main()
