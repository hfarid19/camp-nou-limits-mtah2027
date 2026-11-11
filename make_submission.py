#!/usr/bin/env python3
"""Write Camp Nou Limits' submission CSV: candidates/final_f4sp ("F4sP").

    pip install -r requirements-submission.txt
    python make_submission.py                    # writes evaluation/results/camp_nou_limits.csv
    python evaluation/validate_submission.py evaluation/results/camp_nou_limits.csv

Reads only the competition input (ATD meta + raw tracking, Smart Tagging) through games.py and
possession_pipeline's readers, runs candidates/final_f4sp/pipeline.predict once and writes
frame,label rows. F4sP runs four members, each trained on the 7 DFL + 48 PFF World Cup 2022 games
(the MS-TCN also on the 20 SkillCorner A-League open-data games):
  - the label-smoothed TCN (nn_pff_tcn, tcn_ls.npz), averaged over the input and 4 jittered copies
    of it, with its feature normalisation moved half-way towards the match's own label-free
    feature statistics (candidates/adapt/member_tcn_tta4_a50.py);
  - the MS-TCN at width 192 with label smoothing, fine-tuned with the 20 SkillCorner games added
    (nn_pff, mstcn_c192ls_skcft.npz);
  - the ball radar with stacked LightGBM (ball_radar_pff);
  - the ball-centred possessor-pointer network at 19.3M parameters (nn_pointer,
    ptr_ballc_p20m.npz).
It then decodes them with candidates/mbr_decode: per stretch of play it keeps the candidate label
sequence (the members' hard vote under v6's rule of 1 s smoothing + 1.25 s controlled-possession
fold, the vote under 11 other smoothing/fold rules, or a single member) with the highest expected
spellF1 against samples from an HMM on the members' mean probabilities (fixed seed). Last, every
frame before each half's kick-off is dead. No ground truth, no training data and no network access
are needed: the members load their committed model files (numpy + LightGBM, and PyTorch on the
CPU for the pointer's forward pass).

The pointer's forward pass runs in PyTorch (CPU, float64, candidates/nn_pointer/ptr_torch.py: the
numpy forward's maths op for op, in a worker process that never loads LightGBM) when torch is
installed, else in numpy (ptr_np.py, the reference). MTAH_PTR_BACKEND=numpy forces the numpy path;
both give the same file. Which one ran is printed on stderr.

Numerical libraries (and the torch worker) are pinned to one thread before numpy is imported. That
alone does not make floating-point results machine-independent; what we tested is that the file is
byte-identical (sha256 60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57) on macOS
arm64 (Accelerate), Linux arm64 (OpenBLAS; also from a fresh clone of the repository) and Linux
x86-64 under Rosetta emulation, with the pinned requirements-submission.txt and Python 3.14.6 (the
Linux containers also need the system OpenMP runtime, libgomp1, for LightGBM), with the torch
backend and, when v8 was first built, with the numpy backend. This script always runs
single-threaded; the 2- and 8-thread runs that gave the same file on macOS went through
candidates/final_f4sp/match_run.py (the same pipeline without the thread pin). Native x86-64
hardware was not tested. Run time with torch: 3-10 minutes on one thread of a shared Apple M5 Max
(depending on what else ran), 15 minutes in a Linux arm64 container; with numpy about 54 minutes on
one thread of a heavily shared Apple M5 Max, over 90% of it in the 19.3M-parameter pointer. See
TECHNICAL.md, "Reproduction".
"""

import os

for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "MTAH_TORCH_THREADS"):
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
    """Load candidates/final_f4sp/pipeline.py under its own name, as candidates/bench.py does:
    the vote's member loader clears every module called "pipeline" while it loads the members."""
    path = os.path.join(ROOT, "candidates", "final_f4sp", "pipeline.py")
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
