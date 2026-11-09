#!/usr/bin/env python3
"""PFF FC World Cup 2022 bench: 64 games never trained on, degraded with degrade_v2 (seed 2027).

    python candidates/pff/bench_pff.py prepare [--games ...]   # build the degraded inputs (cache)
    python candidates/pff/bench_pff.py run <pipeline.py> --name NAME [--games G ...] [--half]
    python candidates/pff/bench_pff.py score NAME [NAME ...] [--ref NAME] [--games ...] [--half]
    python candidates/pff/bench_pff.py sweep [--half]          # offline vote x fold sweep (members)

run      loads the candidate ONCE with its FULL models (no PFF game was ever trained on, so no
         leave-one-game-out fold models: BENCH_DFL_FOLD / BENCH_DFL_FOLDS must not be set, and the
         runner refuses to start if they are) and saves its codes for every game as
         $MTAH_WORK/probs/pff/NAME/pff2_<game>_2027_codes.npy. Existing files are skipped, so a run
         can be resumed or extended. One candidate per process (ensembles load modules called
         members / ensemble_core from their own folders).
score    spellF1 per game against the chosen labels (labels.CHOSEN) and against every label
         variant, with fine readings of the chosen labels: raw, control.fold at T = 0.5 and 1 s
         (fine3 = their mean), and T = 1.5 / 2 s for reference (bench.controlled_labels). With
         --ref, paired differences per game: mean, SE over games, wins.
sweep    re-votes saved member codes (temporal_tcn_v2 raw/ctrl x handover_mstcn_v2 raw/ctrl +
         ball_radar) with ensemble_ctrl_v4's machinery over output fold T, offline.

Inputs: pff2_prepared(game, seed) is bench.dfl2_prepared's code with its own cache prefix (pff2_)
and folder ($MTAH_WORK/pff_wc2022/bench_cache): labels from status / poss of the chosen npz
($MTAH_WORK/pff_wc2022/npz), degrade_v2 on the clean players and gap-filled ball, blocks = each
half's labelled rows (kick-off to the period's end). Status is the same in every label variant, so
the degraded inputs are too; variants differ only in the team on live frames.

--half: a fixed random half of the 64 games (numpy seed 2027), for a quicker first pass.
"""

import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [HERE, CAND, ROOT, os.path.join(ROOT, "evaluation")]
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
PFF = os.path.join(WORK, "pff_wc2022")
NPZ = os.path.join(PFF, "npz")
CACHE = os.path.join(PFF, "bench_cache")
PROBS = os.path.join(WORK, "probs", "pff")
SEED = 2027

import possession_metrics as pm   # noqa: E402
import labels as L                # noqa: E402


def all_games():
    return sorted(os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(NPZ, "PFF-*.npz")))


def pick_games(games=None, half=False):
    g = games or all_games()
    if half:
        rng = np.random.default_rng(SEED)
        g = sorted(rng.choice(sorted(g), len(g) // 2, replace=False).tolist())
    return g


def gt_from(status, poss, half):
    gt = np.full(len(status), pm.X, np.int8)
    gt[(status == 1) & (poss == 1)] = pm.H
    gt[(status == 1) & (poss == 2)] = pm.A
    gt[status == 0] = pm.D
    blocks = []
    for h in (1, 2):
        idx = np.where((half == h) & (gt != pm.X))[0]
        if len(idx):
            blocks.append((len(blocks) + 1, int(idx[0]), int(idx[-1])))
    return gt, blocks


def pff2_prepared(game, seed=SEED):
    """bench.dfl2_prepared for a PFF game (same degrade_v2 call), cached as pff2_<game>_<seed>.npz."""
    import degrade_v2
    import tune_on_sample_data as tsd
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "pff2_{}_{}.npz".format(game, seed))
    if not os.path.isfile(path):
        z = np.load(os.path.join(NPZ, "PFF-" + game + ".npz"))
        status = z["status"]
        gt, blocks = gt_from(status, z["poss"], z["half"])
        players, ball = z["players"].astype(np.float64), z["ball"].astype(np.float64)
        team = z["team_of_player"].astype(int)
        slots, slot_team, dball = degrade_v2.degrade_v2(players, team, ball, status, gt != pm.X,
                                                        np.random.default_rng(seed))
        sign = tsd.attack_sign(tsd.Game(game, players, team, ball, []), blocks)
        tmp = path[:-4] + ".tmp{}.npz".format(os.getpid())          # atomic: parallel runners
        np.savez(tmp, slots=slots, slot_team=slot_team, ball=dball, gt=gt,
                 blocks=np.array(blocks), sign=sign)
        os.replace(tmp, path)
    z = np.load(path)
    return (z["slots"], z["slot_team"], z["ball"], z["gt"],
            [tuple(int(v) for v in b) for b in z["blocks"]], float(z["sign"]))


def variant_gt(game, variant):
    z = np.load(os.path.join(PFF, "variants", variant, "PFF-" + game + ".npz"))
    return gt_from(z["status"], z["poss"], z["half"])[0]


# ------------------------------------------------------------------------------------------------

def load_candidate(path, name):
    spec = importlib.util.spec_from_file_location("pffcand_" + name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(module)
    return module


def run(path, name, games):
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS"):
        if k in os.environ:
            raise SystemExit("{} is set: the PFF bench uses full models only".format(k))
    import tune_on_sample_data as tsd
    out = os.path.join(PROBS, name)
    os.makedirs(out, exist_ok=True)
    todo = [g for g in games if not os.path.isfile(os.path.join(out, "pff2_{}_{}_codes.npy".format(g, SEED)))]
    print("{} -> {}: {} games to run ({} done)".format(path, out, len(todo), len(games) - len(todo)), flush=True)
    if not todo:
        return
    module = load_candidate(path, name)
    for g in todo:
        slots, slot_team, ball, gt, blocks, sign = pff2_prepared(g)
        args = (slots, slot_team, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
        t0 = time.time()
        pred = np.asarray(module.predict(*args), np.int8)
        secs = time.time() - t0
        assert pred.shape == gt.shape
        np.save(os.path.join(out, "pff2_{}_{}_codes.npy".format(g, SEED)), pred)
        r = pm.evaluate(gt, pred, bootstrap_n=0)
        print("  {:6s} spellF1 {:.4f}  macroF1 {:.4f}  poss pred/gt {}/{}  {:.1f}s".format(
            g, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], secs), flush=True)
    if "dfl_folds" in sys.modules and sys.modules["dfl_folds"].RESOLVED:
        raise SystemExit("fold files were resolved during a PFF run: {}".format(sys.modules["dfl_folds"].RESOLVED))


# ------------------------------------------------------------------------------------------------

READ = ("raw", "T0.5", "T1", "T1.5", "T2")
FINE = ("raw", "T0.5", "T1")


def targets(game):
    import bench
    gt = pff2_prepared(game)[3]
    t = {"raw": gt}
    for T in (0.5, 1, 1.5, 2):
        t["T{:g}".format(T)] = bench.controlled_labels(gt, T)
    for v in L.VARIANTS:
        t[v] = gt if v == L.CHOSEN else variant_gt(game, v)
    return t


_TCACHE = {}


def score_codes(pred, game):
    if game not in _TCACHE:
        _TCACHE[game] = targets(game)
    T = _TCACHE[game]
    r = {k: pm.evaluate(g, pred, bootstrap_n=0)["spell_f1_mean"] for k, g in T.items()}
    r["fine3"] = float(np.mean([r[k] for k in FINE]))
    ev = pm.evaluate(T["raw"], pred, bootstrap_n=0)
    r["macroF1"] = ev["macro_f1"]
    r["n_pred"], r["n_gt"] = ev["n_pred_spells"], ev["n_gt_spells"]
    return r


def score_dir(name, games):
    rows = {}
    for g in games:
        f = os.path.join(PROBS, name, "pff2_{}_{}_codes.npy".format(g, SEED))
        if os.path.isfile(f):
            rows[g] = score_codes(np.load(f), g)
    return rows


def paired(a, b, key):
    common = sorted(set(a) & set(b))
    d = np.array([a[g][key] - b[g][key] for g in common])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return float(d.mean()), float(se), int((d > 0).sum()), int((d < 0).sum()), len(d)


def cols():
    return READ + ("fine3",) + tuple(v for v in L.VARIANTS if v != L.CHOSEN) + ("macroF1",)


def print_table(results, ref=None):
    c = cols()
    print("labels: chosen = {} (raw, T* = control.fold of it); other columns = other variants raw".format(L.CHOSEN))
    print("{:24s} {:>3s} ".format("candidate", "n") + " ".join("{:>8s}".format(k) for k in c) + "  pred/gt poss")
    for name, rows in results.items():
        m = {k: np.mean([r[k] for r in rows.values()]) for k in c}
        se = np.std([r["raw"] for r in rows.values()], ddof=1) / np.sqrt(max(len(rows), 1)) if len(rows) > 1 else 0
        print("{:24s} {:3d} ".format(name[:24], len(rows)) + " ".join("{:8.4f}".format(m[k]) for k in c)
              + "  {}/{}   (raw SE {:.4f})".format(sum(r["n_pred"] for r in rows.values()),
                                                   sum(r["n_gt"] for r in rows.values()), se))
    if ref:
        print("\npaired vs {} (mean diff ± SE over games, wins/losses):".format(ref))
        for name, rows in results.items():
            if name == ref:
                continue
            parts = []
            for k in ("raw", "T0.5", "T1", "fine3", "T1.5", "T2") + tuple(v for v in L.VARIANTS if v != L.CHOSEN):
                mu, se, w, l, n = paired(rows, results[ref], k)
                parts.append("{} {:+.4f}±{:.4f} ({}/{})".format(k, mu, se, w, n))
            print("  {:22s} ".format(name[:22]) + "  ".join(parts))


# ------------------------------------------------------------------------------------------------

MEMBER_DIRS = {"tcn_raw": "m_tcn_raw", "tcn_ctrl": "m_tcn_ctrl", "ms_raw": "m_mstcn_raw",
               "ms_ctrl": "m_mstcn_ctrl", "radar": "m_ball_radar"}


def sweep(games):
    """Offline: ensemble_ctrl_v4's vote (merge 1 s, fold T) over member training variants."""
    sys.path.insert(0, os.path.join(CAND, "ensemble_ctrl_v4"))
    import control
    import ensemble_core as ec
    import members
    lde = members.load("learned_dfl_evidence")
    combos = {(a, b): [MEMBER_DIRS["tcn_" + a], MEMBER_DIRS["ms_" + b], MEMBER_DIRS["radar"]]
              for a in ("raw", "ctrl") for b in ("raw", "ctrl")}
    FO = (0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    S = {}
    for g in games:
        files = {k: os.path.join(PROBS, d, "pff2_{}_{}_codes.npy".format(g, SEED)) for k, d in MEMBER_DIRS.items()}
        if not all(os.path.isfile(f) for f in files.values()):
            continue
        blocks = pff2_prepared(g)[4]
        runs = [(f, l + 1) for _, f, l in blocks]
        for combo, dirs in combos.items():
            cl = [np.load(os.path.join(PROBS, d, "pff2_{}_{}_codes.npy".format(g, SEED))).astype(np.int8) for d in dirs]
            v = np.asarray(ec.hard_vote(cl, blocks, lde, 25, None, dict(min_poss_s=1.0, min_dead_s=1.0)), np.int8)
            for fo in FO:
                S[(combo, fo, g)] = score_codes(control.fold(v, fo, runs, 25), g)
        for k, d in MEMBER_DIRS.items():
            S[(("member", k), None, g)] = score_codes(np.load(files[k]).astype(np.int8), g)
    gs = sorted({k[2] for k in S})
    print("sweep over {} games: ensemble_ctrl_v4 vote of (TCN variant, MS-TCN variant, ball_radar), merge 1 s, fold T".format(len(gs)))
    keys = ("raw", "T0.5", "T1", "fine3", "T1.5", "T2") + tuple(v for v in L.VARIANTS if v != L.CHOSEN)
    print("{:28s} ".format("config") + " ".join("{:>8s}".format(k) for k in keys) + "  poss/game")
    ref = (("raw", "raw"), 1.25)
    for cfg in sorted({k[:2] for k in S}, key=lambda c: (str(c[0]), c[1] if c[1] is not None else -1)):
        rows = {g: S[cfg + (g,)] for g in gs}
        line = "{:28s} ".format(str(cfg)) + " ".join("{:8.4f}".format(np.mean([r[k] for r in rows.values()])) for k in keys)
        line += "  {:6.0f}".format(np.mean([r["n_pred"] for r in rows.values()]))
        if cfg != ref and cfg[0][0] != "member":
            mu, se, w, l, n = paired(rows, {g: S[ref + (g,)] for g in gs}, "raw")
            line += "   raw vs v4 {:+.4f}±{:.4f} ({}/{})".format(mu, se, w, n)
        print(line)
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["prepare", "run", "score", "sweep"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--name", default=None)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    games = pick_games(a.games, a.half)
    if a.cmd == "prepare":
        for g in games:
            pff2_prepared(g)
    elif a.cmd == "run":
        run(a.args[0], a.name or os.path.basename(os.path.dirname(os.path.abspath(a.args[0]))), games)
    elif a.cmd == "score":
        res = {n: score_dir(n, games) for n in a.args}
        print_table(res, a.ref)
    else:
        sweep(games)


if __name__ == "__main__":
    main()
