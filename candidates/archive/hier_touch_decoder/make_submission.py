#!/usr/bin/env python3
"""Write the competition submission with hier_touch_decoder, without running the dev bench.

    python -I candidates/hier_touch_decoder/make_submission.py [--out PATH]

Reads the match through games.require (the gated data bundle: ./data or $MTAH_DATA_DIR, see
DATA.md), runs pipeline.predict and writes frame,label rows for every tactical-video frame.
Same reading and writing as candidates/bench.py --match, so the file is byte-identical to the
committed submission.csv (md5 in README.md). Needs the pinned requirements.txt; the pipeline
asserts the lightgbm version and the frozen model md5s. Deterministic: no RNG, num_threads=1.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import possession_pipeline as pp  # noqa: E402
import pipeline  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

LABELS = ("H", "A", "D", "X")   # codes 0..3, the possession_metrics convention


def main():
    assert os.environ.get("HTD_SWEEP") != "1", "HTD_SWEEP must be unset when writing the submission"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "submission.csv"))
    args = ap.parse_args()
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n_frames = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n_frames, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    codes = np.asarray(pipeline.predict(players, slot_team, ball, sign_half1, blocks, tags, names,
                                        GAMES[GAME]["pitch"], GAMES[GAME]["fps"]), np.int8)
    assert codes.shape == (n_frames,), codes.shape
    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    letters = np.array(LABELS)[codes]
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))
    graded = np.zeros(n_frames, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    print("wrote {}: {} rows, dead share {:.3f} of graded frames".format(
        args.out, n_frames, float((codes[graded] == 2).mean())))


if __name__ == "__main__":
    main()
