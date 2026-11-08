#!/usr/bin/env python3
"""Write the competition submission CSV with this candidate, and nothing else.

    python -I candidates/learned_dfl_evidence/make_submission.py [--out PATH]

Reads only the competition input (ATD meta + raw tracking, Smart Tagging) through games.py and
possession_pipeline's readers, runs pipeline.predict once and writes frame,label rows (default:
candidates/learned_dfl_evidence/submission.csv). No dev bench, no scratch paths, no ground truth;
the only thing used from evaluation/ is possession_metrics (label letters and summary counts).
The output is byte-identical to `candidates/bench.py ... --match`.
Dependencies: requirements.txt next to this file (lightgbm version asserted by pipeline.py).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm      # noqa: E402
import possession_pipeline as pp     # noqa: E402
import pipeline as lde               # noqa: E402


def main():
    stray = sorted(k for k in os.environ if k.startswith("LDE_"))
    assert not stray, "unset {} before writing the submission (tuning overrides)".format(stray)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "submission.csv"))
    args = ap.parse_args()
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n_frames = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    codes = np.asarray(lde.predict(players, slot_team, ball, sign_half1, blocks, tags, names,
                                   GAMES[GAME]["pitch"], GAMES[GAME]["fps"]), np.int8)
    assert codes.shape == (n_frames,), codes.shape
    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    letters = np.array(pm.LABELS)[codes]
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
