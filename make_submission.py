#!/usr/bin/env python3
"""Write Camp Nou Limits' submission CSV: candidates/final_v11 (v11).

    pip install -r requirements-submission.txt
    python make_submission.py                    # writes evaluation/results/camp_nou_limits.csv
    python evaluation/validate_submission.py evaluation/results/camp_nou_limits.csv

Reads only the competition input (ATD meta + raw tracking, Smart Tagging) through games.py and
possession_pipeline's readers, runs candidates/final_v11/pipeline.predict once and writes
frame,label rows. v11 is v9 (candidates/final_f4spm) with two changes: its TCN and MS-TCN are
retrained on 29 extra input columns, and a stacker joins the decode as a fifth member. The four
models were each trained on the 7 DFL + 48 PFF World Cup 2022 games (the MS-TCN also on the 20
SkillCorner A-League games and the 3 Metrica sample games); the stacker was trained on the 3 Metrica
sample games only:
  - the label-smoothed TCN on 223 inputs (the 194 repository features + the 29 columns of
    candidates/acc_shape/nf_features.py: ball and player acceleration, ball turn angle, team-shape
    hull features), averaged over the input and 4 jittered copies of it, with its feature
    normalisation moved half-way towards the match's own label-free feature statistics
    (candidates/acc_shape/member_tcn_as_tta4_a50.py, tcn_as.npz);
  - the MS-TCN at width 192 with label smoothing on the same 223 inputs, fine-tuned with the 20
    SkillCorner games added and then with the 3 Metrica sample games mixed in
    (candidates/acc_shape/member_mstcn_as_skcft_metft.py, mstcn_as_skcft_metft.npz);
  - the ball radar with stacked LightGBM (ball_radar_pff), as in v9;
  - the ball-centred possessor-pointer network at 19.3M parameters (nn_pointer,
    ptr_ballc_p20m.npz), as in v9;
  - a stacker (candidates/final_v11/stacker_v11.txt, applied by candidates/stacker_met/stacker.py):
    LightGBM, 200 trees per class, on the four models' probabilities at 7 time offsets (84 inputs).
It then decodes the five members with candidates/mbr_decode: per stretch of play it keeps the
candidate label sequence (the members' hard vote under v6's rule of 1 s smoothing + 1.25 s
controlled-possession fold, the vote under 11 other smoothing/fold rules, or a single member) with
the highest expected spellF1 against samples from an HMM on the members' mean probabilities (fixed
seed; the decoder setting is v9's). Last, every frame before each half's kick-off is dead. No ground
truth, no training data and no network access are needed: the members load their committed model
files (numpy + scipy + LightGBM, and PyTorch on the CPU for the pointer's forward pass).

The pointer's forward pass runs in PyTorch (CPU, float64, candidates/nn_pointer/ptr_torch.py: the
numpy forward's maths op for op, in a worker process that never loads LightGBM) when torch is
installed, else in numpy (ptr_np.py, the reference). MTAH_PTR_BACKEND=numpy forces the numpy path.
Which one ran is printed on stderr. The numpy path was not run for v11.

Numerical libraries (and the torch worker) are pinned to one thread before numpy is imported. That
alone does not make floating-point results machine-independent. What has been checked for v11 so
far: the file was regenerated twice on macOS arm64 (Accelerate, torch backend, one thread), through
two drivers (this script, and a driver that runs the same pipeline and logs memory), and both runs
wrote the same file, sha256 9ba4f14d75d872284ecd76cd1c6c6308ad5bf7f9d6c9fedcf3d2b2ce46586938.
PENDING for v11 <!-- TODO final check -->: this script run twice in a row, other thread counts, Linux
arm64 containers, a fresh clone of the repository, and Linux x86-64 under emulation. Those checks
were done for v9 (--candidate final_f4spm: sha256
9926b7c56557a7aedb39597b7c2f52b37c82028d01b394021a360bb3aabcb3e2) and for v8 (--candidate final_f4sp:
60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57) with the pinned
requirements-submission.txt and Python 3.14.6 (the Linux containers also need the system OpenMP
runtime, libgomp1, for LightGBM); they say nothing yet about v11, whose stacker is sensitive to
single float32 steps in its inputs. Native x86-64 hardware was not tested for any version.
Run time of v11 with torch: 9-11 minutes on one thread of an Apple M5 Max that was busy with other
jobs (v9: 2-2.5 minutes on a lightly loaded one); peak memory 5.9 GB. See TECHNICAL.md, section 6
"How to Reproduce", and candidates/final_v11/README.md.

--candidate picks the ensemble folder under candidates/ (default: DEFAULT_CANDIDATE = final_v11,
the submission above). final_f4spm ("F4sP-MF2", v9, the previous submission), final_f4sp ("F4sP",
v8) and final_f4sps ("F4sP + S2", candidate v10: v8 plus a Metrica-trained stacker as a fifth
decode member, never the submission; see its README.md) are opt-in; a non-default candidate needs
an explicit --out, so it never overwrites evaluation/results/camp_nou_limits.csv by accident.
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


DEFAULT_CANDIDATE = "final_v11"
CANDIDATES = ("final_f4sp", "final_f4spm", "final_f4sps", "final_v11")


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
