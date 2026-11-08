#!/usr/bin/env python3
"""ds_v2s: ds_v2 with the attack direction (sign) fixed.

make_dataset_v2 takes `sign` from learned_dfl_evidence.make_dataset.attack_sign, which compares the
home team's mean x over the WHOLE first half with the halfway line. For DFL-MAT-J03WR9 that gives -1
(mean 54.9 m), but at the first-half kick-off the home team stands at x = 45.7 m, i.e. it attacks +x:
the bench (tune_on_sample_data.attack_sign, first 50 frames) uses +1. So ds_v2's J03WR9 features (3 of
21 training degradations) were computed with the attack direction reversed. Here every file is
re-signed with the bench's rule; files whose sign is unchanged are hard-linked, the others' raw npz is
rewritten with the new sign and their ds features recomputed (same slots / ball / idx / labels).

    python -I candidates/fix_attack_sign_ds.py [--check]     # --check: recompute one unchanged file and compare X
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = HERE
ROOT = os.path.dirname(CAND)
sys.path[:0] = [ROOT, CAND, os.path.join(CAND, "learned_dfl_evidence")]
import features as fe            # noqa: E402  (learned_dfl_evidence's, as make_dataset_v2)
import tune_on_sample_data as tsd  # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SRC, DST = os.path.join(WORK, "ds_v2"), os.path.join(WORK, "ds_v2s")
PITCH, FPS = (105.0, 68.0), 25


def bench_sign(match, blocks):
    g = np.load(os.path.join(WORK, "idsse", "npz", match + ".npz"))
    return tsd.attack_sign(tsd.Game(match, g["players"].astype(np.float64), g["team_of_player"].astype(int),
                                    g["ball"].astype(np.float64), []), blocks)


def features(raw, sign, idx):
    fs = fe.frame_sign(len(raw["ball"]), sign, [tuple(int(v) for v in b) for b in raw["blocks"]])
    F, names, kinds = fe.compute(raw["slots"], raw["slot_team"], raw["ball"], fs, PITCH, FPS)
    return F[idx], names, kinds


def main():
    check = "--check" in sys.argv
    for sub in ("ds", "raw"):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)
    files = sorted(f for f in os.listdir(os.path.join(SRC, "raw")) if f.endswith(".npz"))
    for f in files:
        match = f[:-4].rsplit("_", 1)[0]
        raw = dict(np.load(os.path.join(SRC, "raw", f)))
        blocks = [tuple(int(v) for v in b) for b in raw["blocks"]]
        s_new = bench_sign(match, blocks)
        ds = dict(np.load(os.path.join(SRC, "ds", f)))
        if check:
            X, _, _ = features(raw, float(raw["sign"]), ds["idx"])
            print(f, "recompute == stored:", np.array_equal(X, ds["X"], equal_nan=True), flush=True)
            return
        changed = s_new != float(raw["sign"])
        print(f, "sign", float(raw["sign"]), "->", s_new, "recompute" if changed else "link", flush=True)
        for sub in ("ds", "raw"):
            d = os.path.join(DST, sub, f)
            if os.path.exists(d):
                os.remove(d)
        j = os.path.join(SRC, "raw", f[:-4] + ".json")
        if os.path.isfile(j) and not os.path.exists(os.path.join(DST, "raw", f[:-4] + ".json")):
            os.link(j, os.path.join(DST, "raw", f[:-4] + ".json"))
        if not changed:
            os.link(os.path.join(SRC, "raw", f), os.path.join(DST, "raw", f))
            os.link(os.path.join(SRC, "ds", f), os.path.join(DST, "ds", f))
            continue
        raw["sign"] = np.float64(s_new)
        np.savez_compressed(os.path.join(DST, "raw", f), **raw)
        X, names, kinds = features(raw, s_new, ds["idx"])
        assert list(names) == [str(s) for s in ds["names"]]
        ds["X"] = X
        np.savez_compressed(os.path.join(DST, "ds", f), **ds)


if __name__ == "__main__":
    main()
