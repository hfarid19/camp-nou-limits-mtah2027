#!/usr/bin/env python3
"""Write the bench --folds JSON: LDE's verified booster folds + this candidate's CRF folds.

    python -I candidates/pathcrf_dynamic/folds_json.py <crf.npz> <crf folds dir> <out.json>
"""
import hashlib
import json
import os
import sys
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

SCRATCH = WORK


def main():
    crf_path, folds_dir, out = sys.argv[1:4]
    folds = json.load(open(SCRATCH + "/folds_verified.json"))
    sha = hashlib.sha256(open(crf_path, "rb").read()).hexdigest()
    folds[sha[:12]] = [folds_dir, "crf"]
    json.dump(folds, open(out, "w"), indent=1)
    print(out, sha[:12])


if __name__ == "__main__":
    main()
