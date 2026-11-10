#!/usr/bin/env python3
"""Seed ensemble inside one MS-TCN member (round 4, D): pack the same-spec models of several seed runs of
one config into one npz per spec (keys m<i>/..., n_members; tcn_np.Model averages the members' mirror-
averaged softmax outputs, as the TCN `seeds3` test did), so the pack runs through the unchanged
pipeline / pipeline_cfg.py and pff_folds.resolve swaps fold packs exactly as single models.

    python -I candidates/nn_pff/seed_pack.py OUT_DIR SEED_DIR_0 SEED_DIR_1 [SEED_DIR_2 ...]

For every fold_pff_<k>.npz / fold_tcn_<g>.npz / full.npz present in ALL seed dirs: the members must share
`spec` and `names`; the pack's train_games = the union (identical across seeds by construction).
"""
import json
import os
import sys

import numpy as np


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    assert len(dirs) >= 2
    os.makedirs(out, exist_ok=True)
    names = sorted(set.intersection(*[{f for f in os.listdir(d) if f.endswith(".npz") and
                                       (f.startswith("fold_") or f == "full.npz") and ".tmp" not in f} for d in dirs]))
    for f in names:
        zs = [np.load(os.path.join(d, f)) for d in dirs]
        spec = {str(z["spec"]) for z in zs}
        assert len(spec) == 1, (f, spec)
        assert all([str(s) for s in z["names"]] == [str(s) for s in zs[0]["names"]] for z in zs)
        pack = {}
        for i, z in enumerate(zs):
            for k in z.files:
                if k in ("train_games", "spec"):
                    continue
                pack["m{}/{}".format(i, k)] = z[k]
        games = sorted(set().union(*[{str(g) for g in z["train_games"]} for z in zs]))
        pack.update(n_members=np.array(len(zs)), train_games=np.array(games), spec=np.array(spec.pop()),
                    seed_dirs=np.array([os.path.abspath(d) for d in dirs]))
        tmp = os.path.join(out, f[:-4] + ".tmp.npz")
        np.savez(tmp, **pack)
        os.replace(tmp, os.path.join(out, f))
        print(f, len(zs), "members")
    json.dump(dict(seed_dirs=[os.path.abspath(d) for d in dirs], files=names),
              open(os.path.join(out, "pack.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
