#!/usr/bin/env python3
"""Round 4: offline hard votes of saved member codes under v5c's rule, scored against the frozen v6 vote and
judged by the pre-registered rule ($MTAH_WORK/arch/round4_prereg.txt). Cheap: no model is run.

    python candidates/nn_pff/vote_r4.py --set NAME=m1,m2,m3 [--set NAME=sorted:m1,...,m5] [--jobs 8] \
        [--log $MTAH_WORK/arch/round4/votes.jsonl]

Member = a config name; its codes are found by name:
  DFL (bench --dfl2 LOGO codes):  probs/nn_pff_dfl/<m> or probs/nn_asformer_dfl/<m>
  SkillCorner (20 games, full):   probs/skc/<m>
  Metrica dev (2 games x 2 seeds): probs/nn_pff_metrica/<m> or probs/metrica_dev/<m>
  PFF out-of-fold (48 games):     probs/nn_pff/<m> or probs/nn_asformer/<m>  (report only; absent for the
                                  64-game retrains, so a set containing one has no PFF column)
Vote = vote_offline.vote: ensemble_core.hard_vote (tie-break = member order) -> merge_short_v2 1 s ->
control.fold 1.25 s (v5c's rule; prekick_dead is a no-op without tags), = v6_core.Vote. "sorted:" orders the
members by their own DFL fine4 (codes alone, no rule), best first (v6's tie-break logic for 5-7 members).
Reference = v6 = t5c, m5c, radar_pff in that order. Paired: DFL over 21 game-seeds (SE over the 7 games, as
vote_offline.paired_dfl), SkillCorner over 20 games, PFF over 48 games, Metrica = mean difference.
Eligible (prereg): DFL fine4 gain >= +0.005 and >= 2 SE; SkillCorner fine3 not worse than v6 by more than
1 SE; Metrica dev not worse than v6 by more than 0.010. Every evaluated set is appended to --log (the
multiplicity count = distinct set names in it). --rule / --ref: member-level diagnostics (e.g. --rule 0,0
--ref t5c: a member's own codes vs its base), logged separately and never judged against the prereg.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import multiprocessing.pool    # noqa: F401  (stdlib queue before nn_pff/queue.py is on sys.path; run with -I)
import multiprocessing.queues  # noqa: F401

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import vote_offline as vo  # noqa: E402

WORK = vo.WORK
PROBS = os.path.join(WORK, "probs")
RULE = (1.0, 1.25)
REF = ["t5c", "m5c", "radar_pff"]


def _first(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def dirs(m):
    return dict(dfl=_first([os.path.join(PROBS, "nn_pff_dfl", m), os.path.join(PROBS, "nn_asformer_dfl", m)]),
                skc=_first([os.path.join(PROBS, "skc", m)]),
                met=_first([os.path.join(PROBS, "nn_pff_metrica", m), os.path.join(PROBS, "metrica_dev", m)]),
                pff=_first([os.path.join(PROBS, "nn_pff", m), os.path.join(PROBS, "nn_asformer", m)]))


def codes(m, domain, unit):
    d = dirs(m)[domain]
    if d is None:
        raise FileNotFoundError("{}: no {} codes".format(m, domain))
    f = {"dfl": "dfl2_{}_{}_codes.npy".format(*unit) if domain == "dfl" else None,
         "skc": "skc2_{}_2027_codes.npy".format(unit), "met": "dev_{}_{}_codes.npy".format(*unit) if domain == "met" else None,
         "pff": "pff2_{}_2027_codes.npy".format(unit)}[domain]
    return np.load(os.path.join(d, f)).astype(np.int8)


def has_all(m, domain):
    d = dirs(m)[domain]
    if d is None:
        return False
    n = {"dfl": 21, "skc": 20, "met": 4, "pff": 48}[domain]
    return sum(f.endswith("_codes.npy") for f in os.listdir(d)) >= n


def _poss(pred, gt_raw, blocks):
    """(predicted possession spells, truth spells, graded minutes) of one unit: possessions per 90 (review 2026-11-09)."""
    ev = vo.pm.evaluate(gt_raw, pred, bootstrap_n=0)
    return np.array([ev["n_pred_spells"], ev["n_gt_spells"], sum(int(l) - int(f) + 1 for _, f, l in blocks) / 25.0 / 60.0])


def score_set(members, cfg=RULE, domains=("dfl", "skc", "met", "pff")):
    out, poss = {}, {}
    if "dfl" in domains:
        out["dfl"], poss["dfl"] = {}, np.zeros(3)
        for u, (b, tg) in vo.dfl_info().items():
            pred = vo.vote([codes(m, "dfl", u) for m in members], b, *cfg)
            out["dfl"][u] = vo.score_dfl(pred, tg)
            poss["dfl"] += _poss(pred, tg["raw"], b)
    if "skc" in domains:
        out["skc"], poss["skc"] = {}, np.zeros(3)
        for g, (b, tg) in vo.skc_info().items():
            pred = vo.vote([codes(m, "skc", g) for m in members], b, *cfg)
            r = {k: vo.pm.evaluate(tg[k], pred, bootstrap_n=0)["spell_f1_mean"] for k in ("raw", "T0.5", "T1")}
            r["fine3"] = float(np.mean(list(r.values())))
            out["skc"][g] = r
            poss["skc"] += _poss(pred, tg["raw"], b)
    if "met" in domains:
        out["met"], poss["met"] = {}, np.zeros(3)
        for u, (b, gt) in vo.met_info().items():
            pred = vo.vote([codes(m, "met", u) for m in members], b, *cfg)
            out["met"][u] = vo.pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"]
            poss["met"] += _poss(pred, gt, b)
    if "pff" in domains:
        out["pff"], poss["pff"] = {}, np.zeros(3)
        for g, (b, tg) in vo.pff_info().items():
            pred = vo.vote([codes(m, "pff", g) for m in members], b, *cfg)
            out["pff"][g] = vo.score_pff(pred, tg)
            poss["pff"] += _poss(pred, tg["raw"], b)
    out["poss"] = poss
    return out


def alone_dfl(m):
    return float(np.mean([r["fine4"] for r in score_set([m], (0.0, 0.0), ("dfl",))["dfl"].values()]))


def parse(spec):
    name, ms = spec.split("=", 1)
    srt = ms.startswith("sorted:")
    return name, ms[len("sorted:"):].split(",") if srt else ms.split(","), srt


def evaluate(args):
    name, members, srt, ref, rule = args
    t0 = time.time()
    order = None
    if srt:
        order = {m: alone_dfl(m) for m in members}
        members = sorted(members, key=lambda m: -order[m])
    doms = [d for d in ("dfl", "skc", "met", "pff") if all(has_all(m, d) for m in members)]
    S = score_set(members, rule, doms)
    res = dict(name=name, members=members, alone_dfl=order, domains=doms, secs=round(time.time() - t0, 1))
    if "dfl" in S:
        d, se, w, n = vo.paired_dfl(S["dfl"], ref["dfl"], "fine4")
        res.update(dfl=float(np.mean([r["fine4"] for r in S["dfl"].values()])), d_dfl=d, se_dfl=se, w_dfl="{}/{}".format(w, n))
        res["d_sel2"] = vo.paired_dfl({u: {"raw": r["raw"]} for u, r in S["dfl"].items()},
                                       {u: {"raw": r["raw"]} for u, r in ref["dfl"].items()}, "raw")[0]
    if "skc" in S:
        d, se, w, n = vo.paired_pff(S["skc"], ref["skc"], "fine3")
        res.update(skc=float(np.mean([r["fine3"] for r in S["skc"].values()])), d_skc=d, se_skc=se, w_skc="{}/{}".format(w, n))
    if "met" in S:
        res.update(met=float(np.mean(list(S["met"].values()))))
        res["d_met"] = res["met"] - float(np.mean(list(ref["met"].values())))
    if "pff" in S:
        d, se, w, n = vo.paired_pff(S["pff"], ref["pff"], "fine3")
        res.update(pff=float(np.mean([r["fine3"] for r in S["pff"].values()])), d_pff=d, se_pff=se, w_pff="{}/{}".format(w, n))
    for k, (n, ng, mins) in S["poss"].items():      # possessions per 90 min: candidate, v6 (ref), truth
        res["p90_" + k] = round(90.0 * n / mins, 1)
        if k in ref["poss"]:
            res["p90ref_" + k] = round(90.0 * ref["poss"][k][0] / ref["poss"][k][2], 1)
        res["p90gt_" + k] = round(90.0 * ng / mins, 1)
    if all(k in res for k in ("d_dfl", "d_skc", "d_met")):
        res["eligible"] = bool(res["d_dfl"] >= 0.005 and res["d_dfl"] >= 2 * res["se_dfl"]
                               and res["d_skc"] >= -res["se_skc"] and res["d_met"] >= -0.010)
    else:
        res["eligible"] = None          # incomplete: not judged
    return res


def line(r):
    f = lambda k, s: "{:+.4f}±{:.4f} ({})".format(r["d_" + k], r["se_" + k], r["w_" + k]) if "d_" + k in r else "-"
    return "{:22s} DFL {} {} | SKC {} {} | Met {} {} | PFF {} {} | {} | {}".format(
        r["name"], "{:.4f}".format(r["dfl"]) if "dfl" in r else "-", f("dfl", 0),
        "{:.4f}".format(r["skc"]) if "skc" in r else "-", f("skc", 0),
        "{:.4f}".format(r["met"]) if "met" in r else "-", "{:+.4f}".format(r["d_met"]) if "d_met" in r else "",
        "{:.4f}".format(r["pff"]) if "pff" in r else "-", f("pff", 0),
        {True: "ELIGIBLE", False: "no", None: "incomplete"}[r["eligible"]], ",".join(r["members"])) + \
        " | p90 " + " ".join("{} {}/{}/{}".format(k, r.get("p90_" + k), r.get("p90ref_" + k), r.get("p90gt_" + k))
                             for k in ("dfl", "skc", "met", "pff") if "p90_" + k in r) + " (cand/v6/truth)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--sets-file", default=None, help="one NAME=members per line")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--log", default=os.path.join(WORK, "arch", "round4", "votes.jsonl"))
    ap.add_argument("--rule", default="1.0,1.25", help="smooth_s,fold_s; 0,0 = members' own codes (diagnostics)")
    ap.add_argument("--ref", default=",".join(REF), help="reference set (default v6); diagnostics only")
    a = ap.parse_args()
    rule = tuple(float(v) for v in a.rule.split(","))
    if rule != RULE or a.ref != ",".join(REF):
        assert not a.log.endswith("votes.jsonl"), "diagnostic runs (other rule / reference) go to their own --log"
    specs = list(a.set)
    if a.sets_file:
        specs += [l.strip() for l in open(a.sets_file) if l.strip() and not l.startswith("#")]
    # warm the shared caches once (targets), then the reference
    vo.dfl_info(), vo.pff_info(), vo.skc_info(), vo.met_info()
    refm = a.ref.split(",")
    ref = score_set(refm, rule, [d for d in ("dfl", "skc", "met", "pff") if all(has_all(m, d) for m in refm)])
    jobs = [parse(s) + (ref, rule) for s in specs]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for r in pool.imap(evaluate, jobs):
            print(line(r), flush=True)
            with open(a.log, "a") as h:
                h.write(json.dumps(dict(r, time=time.strftime("%Y-%m-%d %H:%M"))) + "\n")


if __name__ == "__main__":
    main()
