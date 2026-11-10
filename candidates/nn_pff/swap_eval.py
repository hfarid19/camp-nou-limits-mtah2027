#!/usr/bin/env python3
"""Offline comparison of member sets under one fixed rule, against a reference member set under the
same rule, on DFL (LOGO codes), PFF (out-of-fold), SkillCorner (20 games; never selection) and
Metrica dev. Members: "dfl_dir|pff_dir|skc_name|metrica_name".

    python candidates/nn_pff/swap_eval.py --rule 1.0,1.25 --ref "m1;m2;m3" --set NAME="m1;m2;m3" ...
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote_offline as vo  # noqa: E402


def scores(members, cfg):
    d, p, s, m = zip(*[x.split("|") for x in members])
    dl = ["{}|{}".format(a, b) for a, b in zip(d, p)]
    S = vo.evaluate_set(dl, [cfg])
    return S["dfl"][cfg], S["pff"][cfg], vo.skc_scores(list(s), cfg), vo.met_scores(list(m), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="1.0,1.25")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--set", action="append", required=True)
    a = ap.parse_args()
    cfg = tuple(float(v) for v in a.rule.split(","))
    R = scores(a.ref.split(";"), cfg)
    for s in [("REF", a.ref)] + [tuple(x.split("=", 1)) for x in a.set]:
        D, P, K, M = scores(s[1].split(";"), cfg)
        line = "{:16s} DFL fine4 {:.4f} {} | PFF fine3 {:.4f} {} | SKC fine3 {:.4f} {} | Metrica {:.4f} ({:+.4f})".format(
            s[0], np.mean([r["fine4"] for r in D.values()]), "{:+.4f}±{:.4f} ({}/{})".format(*vo.paired_dfl(D, R[0], "fine4")),
            np.mean([r["fine3"] for r in P.values()]), "{:+.4f}±{:.4f} ({}/{})".format(*vo.paired_pff(P, R[1], "fine3")),
            np.mean([r["fine3"] for r in K.values()]), "{:+.4f}±{:.4f} ({}/{})".format(*vo.paired_pff(K, R[2], "fine3")),
            np.mean(list(M.values())), np.mean(list(M.values())) - np.mean(list(R[3].values())))
        print(line, flush=True)


if __name__ == "__main__":
    main()
