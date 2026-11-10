#!/usr/bin/env python3
"""All nn_pff / ensemble_ctrl_v6 result tables from saved codes (README numbers come from here).

    python candidates/nn_pff/report.py

For each entry: DFL fine4 (bench --dfl2 LOGO codes, 7 games x 3 seeds), PFF out-of-fold fine3 and
touch spellF1 (48 selection games), Metrica dev spellF1 where codes exist, each paired against
the reference (mean difference, SE over games, wins). The 16-game PFF holdout is never read here
(it was scored once, by the coordinator, for the frozen v6; see ../ensemble_ctrl_v6/README.md).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vote_offline as vo  # noqa: E402

W = vo.WORK
# (label, DFL codes dir under probs/, PFF codes name, Metrica dev codes dir or None)
ENSEMBLES = [
    ("ensemble_ctrl_v5c (reference)", "ensemble_ctrl_v5c", "ensemble_ctrl_v5c", "ensemble_ctrl_v5c"),
    ("ensemble_ctrl_v4", "ensemble_ctrl_v4", "ensemble_ctrl_v4", "ensemble_ctrl_v4"),
    ("v6 members, 0.75 s rule (pipeline_v5cpff.py, live)", "ensemble_ctrl_v6_v5cpff", "ensemble_ctrl_v6_v5cpff", "ensemble_ctrl_v6_v5cpff"),
    ("v6 frozen = members + v5c rule (pipeline.py, live)", "ensemble_ctrl_v6_v5crule", "ensemble_ctrl_v6_v5crule", None),
]
MEMBERS = [
    ("temporal_tcn_v3", "temporal_tcn_v3", "m3_tcn", "v5c_temporal_tcn_v3_pipeline.py"),
    ("handover_mstcn_v3", "handover_mstcn_v3", "m3_mstcn", "v5c_handover_mstcn_v3_pipeline.py"),
    ("ball_radar_v3", "ball_radar_v3", "m3_radar", "v5c_ball_radar_v3_pipeline.py"),
    ("t5c (TCN, v5c+PFF)", "nn_pff_dfl/t5c", "t5c", "t5c"),
    ("m5c (MS-TCN, v5c+PFF)", "nn_pff_dfl/m5c", "m5c", "m5c"),
    ("radar_pff (v5c+PFF)", "nn_pff_dfl/radar_pff", "radar_pff", "radar_pff"),
]
GRID = [("e3_tcn", "TCN base C64 / RF 108 s"), ("gW_tcn", "TCN width C128"), ("gX_tcn", "TCN context RF 415 s, crop 2048"),
        ("gWX_tcn", "TCN width + context"), ("e1_base", "MS-TCN base C64 L10 crop 1024"),
        ("gW_mstcn", "MS-TCN width C128"), ("gX_mstcn", "MS-TCN context L12 crop 4096"),
        ("gWX_mstcn", "MS-TCN width + context")]
AXES = [("t5c", "base (t5c)"), ("t5c_W", "C128 (width)"), ("o_seed1", "seed 1"), ("o_seed2", "seed 2"),
        ("seeds3", "3-seed softmax average (t5c + seeds 1, 2)"),
        ("o_ema", "EMA 0.999"), ("o_focal2", "focal gamma 2"), ("o_transw1", "boundary-weighted CE (trans_w 1)"),
        ("o_cw1", "hand-over loss weight 1.0"), ("o_ls05", "label smoothing 0.05"), ("o_lr05", "lr x0.5"),
        ("o_lr2", "lr x2"), ("o_cos", "cosine schedule"), ("o_wd1e3", "weight decay 1e-3"),
        ("o_wdfl30", "DFL sampling share 0.30"), ("o_ftdfl", "fine-tune on DFL after joint"),
        ("t5c_skc", "+ SkillCorner 15 games"), ("t5c_fp", "+ FOOTPASS 38 games")]
MAXES = [("m5c", "base (m5c)"), ("m5c_W", "C128 / Cr 64 (width)"), ("m5c_ls", "label smoothing 0.05"),
         ("m5c_skc", "+ SkillCorner 15 games")]


def dfl_rows(d):
    p = os.path.join(W, "probs", d)
    if not os.path.isdir(p) or len([f for f in os.listdir(p) if f.startswith("dfl2_") and f.endswith("_codes.npy")]) < 21:
        return None
    return {u: vo.score_dfl(np.load(os.path.join(p, "dfl2_{}_{}_codes.npy".format(*u))).astype(np.int8), tg)
            for u, (b, tg) in vo.dfl_info().items()}


def pff_rows(n, games=None):
    try:
        d = vo.find_pff(n)
    except SystemExit:
        return None
    info = vo.pff_info() if games is None else games
    out = {}
    for g, (b, tg) in info.items():
        f = os.path.join(d, "pff2_{}_2027_codes.npy".format(g))
        if not os.path.isfile(f):
            return None
        out[g] = vo.score_pff(np.load(f).astype(np.int8), tg)
    return out


def met_rows(n):
    if n is None:
        return None
    try:
        return {u: vo.pm.evaluate(gt, vo.met_codes(n, u), bootstrap_n=0)["spell_f1_mean"] for u, (b, gt) in vo.met_info().items()}
    except SystemExit:
        return None


def fmt_pair(mu, se, w, n):
    return "{:+.4f} ± {:.4f} ({}/{})".format(mu, se, w, n)


def table(entries, ref_dfl, ref_pff, title):
    print("\n## " + title)
    print("| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |")
    print("|---|---|---|---|---|---|---|")
    for lab, d, p, m in entries:
        D, P, M = dfl_rows(d) if d else None, pff_rows(p) if p else None, met_rows(m)
        c = ["{:.4f}".format(np.mean([r["fine4"] for r in D.values()])) if D else "-",
             fmt_pair(*vo.paired_dfl(D, ref_dfl, "fine4")) if D and ref_dfl and D is not ref_dfl else "",
             "{:.4f}".format(np.mean([r["fine3"] for r in P.values()])) if P else "-",
             fmt_pair(*vo.paired_pff(P, ref_pff, "fine3")) if P and ref_pff and P is not ref_pff else "",
             "{:.4f}".format(np.mean([r["raw"] for r in P.values()])) if P else "-",
             "{:.4f}".format(np.mean(list(M.values()))) if M else "-"]
        print("| {} | ".format(lab) + " | ".join(c) + " |")


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    rd, rp = dfl_rows("ensemble_ctrl_v5c"), pff_rows("ensemble_ctrl_v5c")
    table(ENSEMBLES, rd, rp, "Ensembles (paired vs ensemble_ctrl_v5c)")
    table(MEMBERS, rd, rp, "Members alone (paired vs ensemble_ctrl_v5c)")
    for title, rows, base in (("Width x context (paired vs the grid base of the same family)", GRID, None),
                              ("Optimisation / loss / data axes on the TCN recipe (paired vs t5c)", AXES, "t5c"),
                              ("Variants of the MS-TCN recipe (paired vs m5c)", MAXES, "m5c")):
        ent = [(lab + " `" + n + "`", "nn_pff_dfl/" + n, n, None) for n, lab in rows]
        if base:
            table(ent, dfl_rows("nn_pff_dfl/" + base), pff_rows(base), title)
        else:
            for fam, b in (("TCN", "e3_tcn"), ("MS-TCN", "e1_base")):
                sub = [e for e in ent if e[0].startswith(fam + " ")]
                table(sub, dfl_rows("nn_pff_dfl/" + b), pff_rows(b), title + ": " + fam)


if __name__ == "__main__":
    main()
