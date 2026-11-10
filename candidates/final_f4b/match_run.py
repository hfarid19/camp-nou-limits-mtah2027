#!/usr/bin/env python3
"""F4b on the BAR-ATH match (Mac only; positions + tags only, as make_submission.py; no ground truth, no scoring).

    OMP_NUM_THREADS=k python candidates/final_f4b/match_run.py --csv <scratch>/f4b_<tag>.csv --tag <tag> [--probes]

Runs the four members one by one (timing each), the MBR decode (pipeline.CFG) + prekick_dead exactly as pipeline.py,
writes the CSV (bench.run_match format) and saves codes + member probabilities to $MTAH_WORK/arch/r5_arch/final_f4b/
match_<tag>.npz (local only). Prints possessions, poss/90, dead share, % of graded frames changed vs v6 (frozen CSV),
vs lsw (ensemble_ctrl_v6/pipeline_lsw.py run 1 CSV) and vs F3 (same members minus the pointer, same decoder), and the
code sha256. --probes: the cross-platform noise probe (member probabilities x (1 + 1e-6 u), u ~ U(-1, 1), 3 seeds) and
the 1e-6 rounding probe.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(CAND, "ensemble_ctrl_v6"),
                os.path.join(CAND, "mbr_decode"), HERE]
import possession_metrics as pm   # noqa: E402
import possession_pipeline as pp  # noqa: E402
import members                    # noqa: E402
import v6_core                    # noqa: E402
import decoder                    # noqa: E402

WORK = os.environ["MTAH_WORK"]
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("final_f4b_pipeline", os.path.join(HERE, "pipeline.py"))
f4b = importlib.util.module_from_spec(_spec)   # unique name: members.load pops plain "pipeline" modules
_spec.loader.exec_module(f4b)
OUT = os.path.join(WORK, "arch", "r5_arch", "final_f4b")


def read_csv_codes(path):
    lab = {l: i for i, l in enumerate(pm.LABELS)}
    rows = open(path).read().splitlines()[1:]
    return np.array([lab[r.split(",")[1]] for r in rows], np.int8)


def decode(probs, codes, blocks, tags, cfg):
    c, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p in probs], codes, blocks, cfg)
    return v6_core.prekick_dead(np.asarray(c, np.int8), tags, blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probes", action="store_true")
    a = ap.parse_args()
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tag_path = require(["smart_tagging"], GAME)[0]
    n = source_frames("tactical", GAME)
    slot_team, _, names, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, n, len(slot_team), GAMES[GAME]["pitch"])
    tags = pp.read_tags(tag_path)
    blocks = half_video_blocks(GAME)
    fps = GAMES[GAME]["fps"]
    args = (players, slot_team, ball, sign_half1, blocks, tags, names, GAMES[GAME]["pitch"], fps)
    t_all = time.time()
    codes, probs = [], []
    for slug, fname in f4b.MEMBERS:
        t = time.time()
        m = members.Member(slug, fname).mod
        codes.append(np.asarray(m.predict(*args), np.int8))
        probs.append(np.asarray(m.predict_proba(*args), np.float32))
        print("member {}/{}: {:.0f}s".format(slug, fname, time.time() - t), flush=True)
    t = time.time()
    final = decode(probs, codes, blocks, tags, f4b.CFG)
    print("decode {:.0f}s; total {:.0f}s".format(time.time() - t, time.time() - t_all), flush=True)
    assert final.shape == (n,) and set(np.unique(final)) <= {0, 1, 2, 3}
    with open(a.csv, "w") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, l) for i, l in enumerate(np.array(pm.LABELS)[final]))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "match_{}.npz".format(a.tag)), final=final, **{"codes{}".format(i): c for i, c in enumerate(codes)},
             **{"probs{}".format(i): p for i, p in enumerate(probs)})
    runs = [(int(f), int(l) + 1) for _, f, l in blocks]
    graded = np.zeros(n, bool)
    for s, e in runs:
        graded[s:e] = True
    min90 = graded.sum() / float(fps) / 60.0 / 90.0
    f3 = decode(probs[:3], codes[:3], blocks, tags, f4b.CFG)
    refs = {"v6": read_csv_codes(os.path.join(ROOT, "evaluation", "results", "camp_nou_limits.csv")),
            "lsw": read_csv_codes(os.path.join(WORK, "arch", "ensemble_ctrl_v6", "submission_lsw_run1.csv")),
            "F3": f3}
    for k, c in [("F4b", final)] + list(refs.items()):
        st = pm.possession_stats(c, runs)
        line = "{:4s} possessions {} ({:.1f}/90), dead share {:.3f}".format(
            k, st["n_spells"], st["n_spells"] / min90, float((c[graded] == 2).mean()))
        if k == "F4b":
            line += ", changed vs " + ", ".join("{} {:.2%}".format(r, float((final != rc)[graded].mean())) for r, rc in refs.items())
            line += ", sha256 {}, csv sha256 {}".format(hashlib.sha256(final.tobytes()).hexdigest()[:16],
                                                       hashlib.sha256(open(a.csv, "rb").read()).hexdigest()[:16])
        print(line, flush=True)
    if a.probes:
        for s in (1, 2, 3):
            rng = np.random.default_rng(s)
            pr = [p.astype(np.float64) * (1 + 1e-6 * rng.uniform(-1, 1, p.shape)) for p in probs]
            c = decode(pr, codes, blocks, tags, f4b.CFG)
            print("noise probe seed {}: frames changed {} ({:.4%} of graded)".format(
                s, int((c != final).sum()), float((c != final)[graded].mean())), flush=True)
        rnd = [np.round(p.astype(np.float64), 6) for p in probs]
        cr = decode(rnd, codes, blocks, tags, f4b.CFG)
        print("rounding probe: rounded-1e-6 vs unrounded: frames changed {}".format(int((cr != final).sum())), flush=True)
        for s in (1, 2, 3):
            rng = np.random.default_rng(s)
            pr = [np.round(p.astype(np.float64) * (1 + 1e-6 * rng.uniform(-1, 1, p.shape)), 6) for p in probs]
            c = decode(pr, codes, blocks, tags, f4b.CFG)
            print("noise then rounding, seed {}: frames changed vs rounded {}".format(s, int((c != cr).sum())), flush=True)


if __name__ == "__main__":
    main()
