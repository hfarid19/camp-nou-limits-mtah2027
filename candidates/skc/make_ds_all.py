#!/usr/bin/env python3
"""Round-5 "+SKC" training files for ALL 20 SkillCorner games (prereg addendum 2026-11-10 11:20: SkillCorner is
cross-validated over 4 folds of 5 games, the 5 holdout games being the 4th fold), in ds_pff2's degradation recipe.

    python -I candidates/skc/make_ds_all.py --allow-skc-holdout [--seeds 41 42 43] [--jobs 8]

Writes $MTAH_WORK/skc_aleague/ds_all/{ds,raw}/SKC-<game>_<seed>.npz with nn_pff/make_dataset.job (as make_ds.py:
same randomised degrade_v2 draws for seeds 41 / 42 / 43, AR(1) ball offset, features, stride-5 phase = seed % 5).
Files that already exist in $MTAH_WORK/skc_aleague/ds (train games, seeds 41 / 42, built by make_ds.py) are
hard-linked, not rebuilt. make_ds.py itself (train games only, holdout refused) is unchanged; this script needs the
explicit --allow-skc-holdout, and the nn trainers accept the holdout games only with NN_PFF_SKC_CV=1 (nn_pff/data.py).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_ds  # noqa: E402

WORK = make_ds.WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-skc-holdout", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    assert a.allow_skc_holdout, "--allow-skc-holdout is required (the 5 SKC holdout games are built too)"
    assert 44 not in a.seeds, "seed 44 is degrade_v4: ds_pff2 is degrade_v2 only"
    base = os.path.join(WORK, "skc_aleague")
    hold = json.load(open(os.path.join(base, "holdout_games.json")))
    games = sorted(hold["train"] + hold["holdout"])
    assert len(games) == 20
    out = os.path.join(base, "ds_all")
    for sub in ("ds", "raw"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    jobs = []
    for g in games:
        for s in a.seeds:
            done = True
            for sub in ("ds", "raw"):
                src = os.path.join(base, "ds", sub, "{}_{}.npz".format(g, s))
                dst = os.path.join(out, sub, "{}_{}.npz".format(g, s))
                if os.path.exists(dst):
                    continue
                if os.path.exists(src):
                    os.link(src, dst)
                    if sub == "raw" and os.path.exists(src[:-4] + ".json"):
                        os.link(src[:-4] + ".json", dst[:-4] + ".json")
                else:
                    done = False
            if not done:
                jobs.append((os.path.join(base, "npz", g + ".npz"), g, s, out))
    with open(os.path.join(out, "config.json"), "w") as h:
        json.dump(dict(dataset="skc_aleague", games=games, seeds=a.seeds, builder="candidates/nn_pff/make_dataset.job",
                       linked_from="skc_aleague/ds", note="all 20 games incl. the 5 holdout (round-5 +SKC CV)"), h,
                  indent=1)
    print("{} files to build".format(len(jobs)), flush=True)
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(make_ds._job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
