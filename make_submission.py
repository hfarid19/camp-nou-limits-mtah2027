#!/usr/bin/env python3
"""Write Camp Nou Limits' submission CSV: candidates/final_v13 (v13).

    pip install -r requirements-submission.txt
    python make_submission.py                    # writes evaluation/results/camp_nou_limits.csv
    python evaluation/validate_submission.py evaluation/results/camp_nou_limits.csv

Reads only the competition input (ATD meta + raw tracking, Smart Tagging) through games.py and
possession_pipeline's readers, runs candidates/final_v13/pipeline.predict once and writes
frame,label rows. v13 is v12 (candidates/final_v12) with unequal member weights in the decode: 1 3 1 2 3
for TCN, MS-TCN, radar, pointer, stacker (a weight w = the member entered w times). Nothing was
retrained. The weights were chosen by the team on 14 Nov 2026 after seeing a search over 1,023
weightings, and they did not pass the rule written before that search
(candidates/final_v13/README.md). v12 is v11 (candidates/final_v11) with its TCN and MS-TCN retrained on 58 more
input columns and its stacker refitted on them; v11 is v9 (candidates/final_f4spm) with the TCN and
MS-TCN retrained on 29 extra input columns and a stacker as a fifth decode member. The four models
were each trained on the 7 DFL + 48 PFF World Cup 2022 games (the MS-TCN also on the 20 SkillCorner
A-League games and the 3 Metrica sample games); the stacker was trained on the 3 Metrica sample games
only:
  - the label-smoothed TCN on 281 inputs (the 194 repository features + the 87 columns of
    candidates/newfeat_v12: v11's ball and player acceleration, ball turn angle and team-shape hull
    columns, plus relative movement of the ball and the nearest players, detected touches, finer ball
    kinematics and hand-picked products), averaged over the input and 4 jittered copies of it, with
    its feature normalisation moved half-way towards the match's own label-free feature statistics
    (candidates/newfeat_v12/member_tcn_v12_tta4_a50.py, tcn_v12.npz);
  - the MS-TCN at width 192 with label smoothing on the same 281 inputs, fine-tuned with the 20
    SkillCorner games added and then with the 3 Metrica sample games mixed in
    (candidates/newfeat_v12/member_mstcn_v12_skcft_metft.py, mstcn_v12_skcft_metft.npz);
  - the ball radar with stacked LightGBM (ball_radar_pff), as in v9;
  - the ball-centred possessor-pointer network at 19.3M parameters (nn_pointer,
    ptr_ballc_p20m.npz), as in v9;
  - a stacker (candidates/final_v12/stacker_v12.txt, applied by candidates/stacker_met/stacker.py):
    LightGBM, 200 trees per class, on the four models' probabilities at 7 time offsets (84 inputs).
It then decodes the five members, weighted as above, with candidates/mbr_decode: per stretch of play it keeps the
candidate label sequence (the members' hard vote under v6's rule of 1 s smoothing + 1.25 s
controlled-possession fold, the vote under 11 other smoothing/fold rules, or a single member) with
the highest expected spellF1 against samples from an HMM on the members' mean probabilities (fixed
seed; the decoder setting is v9's). Last, every frame before each half's kick-off is dead. No ground
truth, no training data and no network access are needed: the members load their committed model
files (numpy + scipy + LightGBM, and PyTorch on the CPU for the pointer's forward pass).

The pointer's forward pass runs in PyTorch (CPU, float64, candidates/nn_pointer/ptr_torch.py: the
numpy forward's maths op for op, in a worker process that never loads LightGBM) when torch is
installed, else in numpy (ptr_np.py, the reference). MTAH_PTR_BACKEND=numpy forces the numpy path.
Which one ran is printed on stderr. For v13 the numpy path wrote the same file as the torch path (about 20 minutes).

Numerical libraries (and the torch worker) are pinned to one thread before numpy is imported. That
alone does not make floating-point results machine-independent. What was checked for v13 on 14 Nov 2026, with the pinned
requirements-submission.txt and Python 3.14.6: every run below wrote the same file, sha256
697fae9f6c9dfb832fadf0ead231be1db3592b1b078b2e56fd5a28ece4215487:
  - this script twice in a row on macOS arm64 (Accelerate, torch backend, one thread);
  - the same steps at 1, 2 and 8 threads, through a driver outside the repository that skips the
    pinning above;
  - a fresh clone of the previous commit plus v13's two code files (this file and
    candidates/final_v13/pipeline.py; only this docstring changed afterwards), in a new environment;
  - that clone in a python:3.14.6-slim Linux arm64 container (OpenBLAS; the container also
    needs the system OpenMP runtime, libgomp1, for LightGBM);
  - the same in a Linux x86-64 container under Rosetta emulation;
  - the numpy pointer backend on macOS arm64.
Not checked: native x86-64 hardware (for any version). v12 (--candidate final_v12: sha256
eb249aa50cc9ab989514e982b9bfefabc3143f52b996811bb209f9429f5d849f), v9 (--candidate final_f4spm:
9926b7c56557a7aedb39597b7c2f52b37c82028d01b394021a360bb3aabcb3e2) and v8 (--candidate final_f4sp:
60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57) had the same kind of checks; no
platform check was run for v11.
Run time of v13 with torch: 3.9 to 4.4 minutes on one thread of an Apple M5 Max (10 minutes in the
Linux arm64 container, 19 minutes under x86-64 emulation); peak memory of the main process 6 GB
(the torch worker is a separate process). See TECHNICAL.md, section 6 "How to Reproduce", and
candidates/final_v13/README.md.

--candidate picks the ensemble folder under candidates/ (default: DEFAULT_CANDIDATE = final_v13,
the submission above). final_v12 (v12, published on 14 Nov 2026 and replaced the same day), final_v11 (v11, published on 14 Nov 2026 and then replaced; sha256
9ba4f14d75d872284ecd76cd1c6c6308ad5bf7f9d6c9fedcf3d2b2ce46586938), final_f4spm ("F4sP-MF2", v9),
final_f4sp ("F4sP", v8) and final_f4sps ("F4sP + S2", candidate v10: v8 plus a Metrica-trained
stacker as a fifth decode member, never the submission; see its README.md) are opt-in; a
non-default candidate needs an explicit --out, so it never overwrites
evaluation/results/camp_nou_limits.csv by accident.
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


DEFAULT_CANDIDATE = "final_v13"
CANDIDATES = ("final_f4sp", "final_f4spm", "final_f4sps", "final_v11", "final_v12", "final_v13")


def load_ensemble(candidate=DEFAULT_CANDIDATE):
    """Load candidates/<candidate>/pipeline.py under its own name, as candidates/bench.py does:
    the vote's member loader clears every module called "pipeline" while it loads the members."""
    path = os.path.join(ROOT, "candidates", candidate, "pipeline.py")
    spec = importlib.util.spec_from_file_location("camp_nou_limits_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="default for the default candidate: evaluation/results/camp_nou_limits.csv")
    ap.add_argument("--candidate", choices=CANDIDATES, default=DEFAULT_CANDIDATE)
    args = ap.parse_args()
    if args.out is None:
        if args.candidate != DEFAULT_CANDIDATE:
            ap.error("--candidate {} is not the submission: give --out explicitly".format(args.candidate))
        args.out = os.path.join(ROOT, "evaluation", "results", "camp_nou_limits.csv")
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n_frames = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    ensemble = load_ensemble(args.candidate)
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
