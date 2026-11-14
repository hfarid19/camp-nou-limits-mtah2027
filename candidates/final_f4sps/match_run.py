#!/usr/bin/env python3
"""F4sP + S2 (candidate v10) on the BAR-ATH match (Mac only; positions + tags only, as make_submission.py; no ground
truth, no scoring).

    OMP_NUM_THREADS=k python -I candidates/final_f4sps/match_run.py --tag <tag> [--csv <path>] [--probes]

Runs the four members one by one (timing each), the stacker, the 5-member MBR decode (pipeline.CFG) + prekick_dead exactly
as pipeline.py, writes the CSV (default $MTAH_WORK/arch/r5_arch/final_f4sps/csv/f4sps_<tag>.csv; refuses any path under
evaluation/) and saves codes + member and stacker probabilities to $MTAH_WORK/arch/r5_arch/final_f4sps/match_<tag>.npz
(local only). Prints possessions, poss/90, mean length, dead share, the possession-length distribution, and the share of
graded frames that differ from v6, v7 (F4b), v8 (F4sP) and v9 (F4sP-MF2) (their CSVs, kept in $MTAH_WORK and checked by
sha256), and the CSV sha256. When v8's saved match outputs (final_f4sp/match_run1.npz) are present it also checks that the
four members' outputs equal v8's and that decoding them without the stacker gives v8's CSV back (so the stacker is the
only difference).
--probes: (1) noise: member probabilities x (1 + 1e-6 u), u ~ U(-1, 1), 3 seeds, then float32, stacker and decode
recomputed; (2) rounding: member probabilities rounded to 1e-6; (3) noise then rounding. For each: frames whose stacker
codes change, largest stacker |dP|, and final frames changed. Also the stacker's threshold margin: for every stacker input
value at the match, its relative distance to the nearest split threshold on that feature in the model.
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
                os.path.join(CAND, "mbr_decode"), os.path.join(CAND, "stacker_met"), HERE]
import possession_metrics as pm   # noqa: E402
import possession_pipeline as pp  # noqa: E402
import members                    # noqa: E402
import v6_core                    # noqa: E402
import decoder                    # noqa: E402
import stacker                    # noqa: E402

WORK = os.environ["MTAH_WORK"]
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("final_f4sps_pipeline", os.path.join(HERE, "pipeline.py"))
f4sps = importlib.util.module_from_spec(_spec)   # unique name: members.load pops plain "pipeline" modules
_spec.loader.exec_module(f4sps)
OUT = os.path.join(WORK, "arch", "r5_arch", "final_f4sps")
REF_CSV = {"v6": (os.path.join(WORK, "arch", "ensemble_ctrl_v6", "submission_frozen_backup.csv"),
                  "da85721c39520c30fbac26c048b4ed032d113a367d4b62653d484c323de0ef26"),
           "v7": (os.path.join(WORK, "arch", "r5_arch", "final_f4b", "fresh2", "out", "fresh.csv"),
                  "228fe6a47e29a411746a97e9c7f758ad557c2f6acf768343b39932ede8ac4f30"),
           "v8": (os.path.join(WORK, "arch", "r5_arch", "final_f4sp", "csv", "f4sp_make_submission.csv"),
                  "60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57"),
           "v9": (os.path.join(WORK, "arch", "r5_arch", "final_f4spm", "csv", "f4spm_torch_t1.csv"),
                  "9926b7c56557a7aedb39597b7c2f52b37c82028d01b394021a360bb3aabcb3e2")}
V8_MATCH = os.path.join(WORK, "arch", "r5_arch", "final_f4sp", "match_run1.npz")


def read_csv_codes(path, sha):
    data = open(path, "rb").read()
    assert hashlib.sha256(data).hexdigest() == sha, path
    lab = {l: i for i, l in enumerate(pm.LABELS)}
    rows = data.decode().splitlines()[1:]
    return np.array([lab[r.split(",")[1]] for r in rows], np.int8)


def decode(probs, codes, blocks, tags, cfg):
    c, _ = decoder.decode([np.asarray(p, np.float32).astype(np.float64) for p in probs], codes, blocks, cfg)
    return v6_core.prekick_dead(np.asarray(c, np.int8), tags, blocks)


def stack_decode(mprobs, mcodes, blocks, tags, booster):
    """The pipeline's path from the four members' outputs: float32 -> stacker -> 5-member decode -> prekick_dead."""
    mp = [np.asarray(p, np.float32).astype(np.float64) for p in mprobs]
    sp, sc = stacker.predict(mp, blocks, booster)
    return decode(mp + [sp], list(mcodes) + [sc], blocks, tags, f4sps.CFG), sp, sc


def spells(c, runs):
    out = []
    for s, e in runs:
        w = c[s:e]
        i = 0
        while i < len(w):
            j = i
            while j < len(w) and w[j] == w[i]:
                j += 1
            out.append((s + i, s + j, int(w[i])))
            i = j
    return out


def length_stats(c, runs, fps, min90):
    """Possession-length distribution (per 90 graded minutes) and quick changes (see README)."""
    sp = spells(c, runs)
    L = np.array([(e - s) / float(fps) for s, e, k in sp if k < 2])
    r = {"<{}s".format(t): float((L < t).sum() / min90) for t in (1, 2, 3)}
    q_aba = q_chg = 0
    for s, e in runs:
        ss = [x for x in sp if s <= x[0] < e]
        q_aba += sum(1 for a, x, y in zip(ss, ss[1:], ss[2:])        # A-B-A, all three possession spells, B < 2 s
                     if a[2] < 2 and x[2] < 2 and a[2] == y[2] != x[2] and (x[1] - x[0]) / float(fps) < 2)
        pos = [x for x in ss if x[2] < 2]
        ch = [x[0] for a, x in zip(pos, pos[1:]) if a[2] != x[2]]    # team changes (dead ball in between allowed)
        q_chg += sum(1 for u, v in zip(ch, ch[1:]) if (v - u) / float(fps) < 2)
    r["ABA<2s"] = q_aba / min90
    r["chg<2s"] = q_chg / min90
    return r


def margin_report(booster, mprobs, blocks):
    """Relative distance of every stacker input value at the match to the nearest split threshold on its feature."""
    model = booster.dump_model()
    th = [[] for _ in range(84)]

    def walk(nd):
        if "split_feature" in nd:
            th[nd["split_feature"]].append(nd["threshold"])
            walk(nd["left_child"])
            walk(nd["right_child"])
    for t in model["tree_info"]:
        walk(t["tree_structure"])
    mp = [np.asarray(p, np.float32).astype(np.float64) for p in mprobs]
    runs = stacker.runs_of(blocks)
    n = len(mp[0])
    ins = np.where(stacker.inside(n, runs))[0]
    X = stacker.feats(mp, runs, ins).astype(np.float64)
    rel = np.full(X.shape, np.inf)
    for f in range(84):
        if not th[f]:
            continue
        t = np.unique(np.array(th[f]))
        i = np.clip(np.searchsorted(t, X[:, f]), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(X), int)
        d = np.minimum(np.abs(X[:, f] - t[i]), np.abs(X[:, f] - t[np.maximum(i - 1, 0)]))
        rel[:, f] = d / np.maximum(np.abs(X[:, f]), 1e-30)
    m = rel.min(1)
    print("stacker threshold margin at the match ({} frames x 84 inputs; {} distinct thresholds): min relative distance "
          "{:.2e}; inputs within 1e-6 relative of a threshold: {} ({} frames); within 1e-5: {} ({} frames); within 1e-7: {}"
          .format(len(X), sum(len(set(x)) for x in th), float(m.min()), int((rel < 1e-6).sum()), int((m < 1e-6).sum()),
                  int((rel < 1e-5).sum()), int((m < 1e-5).sum()), int((rel < 1e-7).sum())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--probes", action="store_true")
    a = ap.parse_args()
    csv_path = a.csv or os.path.join(OUT, "csv", "f4sps_{}.csv".format(a.tag))
    if os.path.abspath(csv_path).startswith(os.path.join(ROOT, "evaluation") + os.sep):
        raise SystemExit("refusing to write the v10 candidate CSV under evaluation/ (v9 stays the submission)")
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
    for slug, fname in f4sps.MEMBERS:
        t = time.time()
        m = members.Member(slug, fname).mod
        codes.append(np.asarray(m.predict(*args), np.int8))
        probs.append(np.asarray(m.predict_proba(*args), np.float32))
        print("member {}/{}: {:.0f}s".format(slug, fname, time.time() - t), flush=True)
    t = time.time()
    booster = stacker.load()
    final, sp, sc = stack_decode(probs, codes, blocks, tags, booster)
    print("stacker + decode {:.1f}s; total {:.0f}s".format(time.time() - t, time.time() - t_all), flush=True)
    assert final.shape == (n,) and set(np.unique(final)) <= {0, 1, 2, 3}
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, l) for i, l in enumerate(np.array(pm.LABELS)[final]))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "match_{}.npz".format(a.tag)), final=final, stack_probs=sp, stack_codes=sc,
             **{"codes{}".format(i): c for i, c in enumerate(codes)}, **{"probs{}".format(i): p for i, p in enumerate(probs)})
    runs = [(int(f), int(l) + 1) for _, f, l in blocks]
    graded = np.zeros(n, bool)
    for s, e in runs:
        graded[s:e] = True
    min90 = graded.sum() / float(fps) / 60.0 / 90.0
    refs = {k: read_csv_codes(p, s) for k, (p, s) in REF_CSV.items()}
    if os.path.isfile(V8_MATCH):
        z = np.load(V8_MATCH)
        same = [i for i in range(4) if np.array_equal(z["probs{}".format(i)], probs[i])
                and np.array_equal(z["codes{}".format(i)], codes[i])]
        v8_re = decode(probs, codes, blocks, tags, f4sps.CFG)
        print("v8 check: members {} equal v8's saved match outputs; this run's four members decoded without the stacker "
              "give v8's CSV: {}".format(same, np.array_equal(v8_re, refs["v8"])), flush=True)
    print("stacker at the match: argmax label changes inside the halves {}".format(
        int(sum((np.diff(sc[s:e]) != 0).sum() for s, e in runs))), flush=True)
    for k, c in [("v10", final)] + list(refs.items()):
        st = pm.possession_stats(c, runs)
        ls = length_stats(c, runs, fps, min90)
        line = "{:4s} possessions {} ({:.1f}/90), mean {:.1f} s, dead share {:.3f}; per 90: <1 s {:.1f}, <2 s {:.1f}, " \
               "<3 s {:.1f}, A-B-A with B < 2 s {:.1f}, team changes < 2 s apart {:.1f}".format(
                   k, st["n_spells"], st["n_spells"] / min90, st["mean_duration_s"], float((c[graded] == 2).mean()),
                   ls["<1s"], ls["<2s"], ls["<3s"], ls["ABA<2s"], ls["chg<2s"])
        if k == "v10":
            line += ", X in halves {}, changed vs ".format(int((c[graded] == 3).sum())) + ", ".join(
                "{} {:.2%} of graded ({} graded frames, {} in all)".format(
                    r, float((final != rc)[graded].mean()), int((final != rc)[graded].sum()), int((final != rc).sum()))
                for r, rc in refs.items())
            line += ", sha256 {}, csv sha256 {}".format(hashlib.sha256(final.tobytes()).hexdigest()[:16],
                                                       hashlib.sha256(open(csv_path, "rb").read()).hexdigest())
        print(line, flush=True)
    if a.probes:
        margin_report(booster, probs, blocks)

        def probe(label, pr, ref_final, ref_sp, ref_sc):
            c, sp2, sc2 = stack_decode(pr, codes, blocks, tags, booster)
            print("{}: stacker codes changed {} frames, stacker max|dP| {:.2e}; final frames changed {} ({:.4%} of graded)"
                  .format(label, int((sc2 != ref_sc).sum()), float(np.abs(sp2.astype(np.float64) - ref_sp).max()),
                          int((c != ref_final).sum()), float((c != ref_final)[graded].mean())), flush=True)
            return c, sp2, sc2
        spd = sp.astype(np.float64)
        for s in (1, 2, 3):
            rng = np.random.default_rng(s)
            probe("noise probe seed {}".format(s),
                  [p.astype(np.float64) * (1 + 1e-6 * rng.uniform(-1, 1, p.shape)) for p in probs], final, spd, sc)
        cr, spr, scr = probe("rounding probe (1e-6) vs unrounded", [np.round(p.astype(np.float64), 6) for p in probs],
                             final, spd, sc)
        for s in (1, 2, 3):
            rng = np.random.default_rng(s)
            probe("noise then rounding, seed {}, vs rounded".format(s),
                  [np.round(p.astype(np.float64) * (1 + 1e-6 * rng.uniform(-1, 1, p.shape)), 6) for p in probs],
                  cr, spr.astype(np.float64), scr)


if __name__ == "__main__":
    main()
