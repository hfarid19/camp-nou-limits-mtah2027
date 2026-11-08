#!/usr/bin/env python3
"""Pack the 7 leave-one-game-out fold models into the shipped model_tcn.npz (their outputs are
averaged at inference) and write the bench folds JSON mapping its sha256 to the fold files.

    python -I pack_model.py FOLDS_DIR [FOLDS_JSON [OUT_NPZ]]

handover_mstcn_v2: OUT_NPZ (default <this folder>/model_tcn.npz) added so each training-target
variant can be packed to its own file (model_tcn_raw.npz, model_tcn_ctrl.npz).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    folds_dir = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(HERE, "model_tcn.npz")
    arrs = {"n_members": np.array(7)}
    for gi in range(7):
        z = np.load(os.path.join(folds_dir, "fold_tcn_{}.npz".format(gi)))
        for k in z.files:
            arrs["m{}/{}".format(gi, k)] = z[k]
    with open(out, "wb") as handle:     # np.savez (no compression): byte-stable for equal arrays
        np.savez(handle, **arrs)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(os.path.basename(out), "sha256", sha)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as handle:
            json.dump({sha[:12]: [folds_dir, "tcn"]}, handle, indent=1)
        print("folds json ->", sys.argv[2])


if __name__ == "__main__":
    main()
