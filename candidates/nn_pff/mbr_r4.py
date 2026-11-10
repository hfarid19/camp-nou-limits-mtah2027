#!/usr/bin/env python3
"""Round 4 members under the fixed MBR decoder of candidates/mbr_decode/pipeline_mbr.py (CFG unchanged, no search).

    python -I candidates/nn_pff/mbr_r4.py --set NAME=tcn,mstcn,radar [--set ...] [--jobs 6] \
        [--log $MTAH_WORK/arch/round4/votes.jsonl]

Members are config names whose saved probabilities + codes exist (mbr_decode/data.py layout), or a folder under
probs/ with every domain's files (contains "/", e.g. adapt32/o_ls05__tta4 = R5's TTA TCN of pipeline_mbr_tta4.py):
  DFL  probs/nn_pff_dfl/<m>/dfl2_<match>_<seed>[_codes].npy   (LOGO)
  SKC  probs/skc_mbr/<m>/skc2_<g>_2027[_codes].npy              (mbr_decode/skc_probs.py)
  Met  probs/nn_pff_metrica/<m>/dev_<g>_<seed>[_codes].npy
  PFF  probs/nn_pff/<m>/pff2_<g>_2027[_codes].npy               (out-of-fold; report only)
Decode = decoder.decode(probs, codes, blocks, pipeline_mbr.CFG) (prekick_dead is a no-op offline), scored with
vote_r4's scorers; paired vs the v6 hard vote (v5c rule) exactly as vote_r4.py, so the prereg flags are
comparable; also paired vs MBR on the lsw members (the round-5 leader). Rows go to --log with decoder="mbr".
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import multiprocessing.pool    # noqa: F401
import multiprocessing.queues  # noqa: F401

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MBR = os.path.join(os.path.dirname(HERE), "mbr_decode")
sys.path.append(HERE)
import vote_r4 as vr  # noqa: E402
vo = vr.vo
sys.path.insert(0, MBR)
import decoder  # noqa: E402


def _cfg():
    src = open(os.path.join(MBR, "pipeline_mbr.py")).read()
    line = [l for l in src.splitlines() if l.startswith("CFG = ")][0]
    return eval(line.split("=", 1)[1].split("#")[0])   # dict(src=..., ...) literal from the frozen file


CFG = _cfg()
P = vr.PROBS


def _path(domain, m, u):
    if "/" in m:     # a folder under probs/ holding every domain's files (e.g. adapt32/o_ls05__tta4: R5's TTA TCN)
        f = {"dfl": "dfl2_{}_{}", "skc": "skc2_{}_2027", "met": "dev_{}_{}", "pff": "pff2_{}_2027"}[domain]
        return os.path.join(P, m, f.format(*u) if isinstance(u, tuple) else f.format(u))
    return {"dfl": os.path.join(P, "nn_pff_dfl", m, "dfl2_{}_{}".format(*u) if domain == "dfl" else ""),
            "skc": os.path.join(P, "skc_mbr", m, "skc2_{}_2027".format(u)),
            "met": os.path.join(P, "nn_pff_metrica", m, "dev_{}_{}".format(*u) if domain == "met" else ""),
            "pff": os.path.join(P, "nn_pff", m, "pff2_{}_2027".format(u))}[domain]


def has(domain, members):
    info = {"dfl": vo.dfl_info, "skc": vo.skc_info, "met": vo.met_info, "pff": vo.pff_info}[domain]()
    return all(os.path.isfile(_path(domain, m, u) + ".npy") and os.path.isfile(_path(domain, m, u) + "_codes.npy")
               for m in members for u in info)


def mbr_pred(domain, members, u, blocks):
    probs = [np.load(_path(domain, m, u) + ".npy").astype(np.float32).astype(np.float64) for m in members]
    codes = [np.load(_path(domain, m, u) + "_codes.npy").astype(np.int8) for m in members]
    return np.asarray(decoder.decode(probs, codes, blocks, CFG)[0], np.int8)


def score_mbr(members, domains):
    out, poss = {}, {}
    if "dfl" in domains:
        out["dfl"], poss["dfl"] = {}, np.zeros(3)
        for u, (b, tg) in vo.dfl_info().items():
            pred = mbr_pred("dfl", members, u, b)
            out["dfl"][u] = vo.score_dfl(pred, tg)
            poss["dfl"] += vr._poss(pred, tg["raw"], b)
    if "skc" in domains:
        out["skc"], poss["skc"] = {}, np.zeros(3)
        for g, (b, tg) in vo.skc_info().items():
            pred = mbr_pred("skc", members, g, b)
            r = {k: vo.pm.evaluate(tg[k], pred, bootstrap_n=0)["spell_f1_mean"] for k in ("raw", "T0.5", "T1")}
            r["fine3"] = float(np.mean(list(r.values())))
            out["skc"][g] = r
            poss["skc"] += vr._poss(pred, tg["raw"], b)
    if "met" in domains:
        out["met"], poss["met"] = {}, np.zeros(3)
        for u, (b, gt) in vo.met_info().items():
            pred = mbr_pred("met", members, u, b)
            out["met"][u] = vo.pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"]
            poss["met"] += vr._poss(pred, gt, b)
    if "pff" in domains:
        out["pff"], poss["pff"] = {}, np.zeros(3)
        for g, (b, tg) in vo.pff_info().items():
            pred = mbr_pred("pff", members, g, b)
            out["pff"][g] = vo.score_pff(pred, tg)
            poss["pff"] += vr._poss(pred, tg["raw"], b)
    out["poss"] = poss
    return out


def paired_vs(res, S, ref, tag):
    if "dfl" in S and "dfl" in ref:
        d, se, w, n = vo.paired_dfl(S["dfl"], ref["dfl"], "fine4")
        res.update({"d_dfl" + tag: d, "se_dfl" + tag: se, "w_dfl" + tag: "{}/{}".format(w, n)})
    for k in ("skc", "pff"):
        if k in S and k in ref:
            d, se, w, n = vo.paired_pff(S[k], ref[k], "fine3")
            res.update({"d_{}{}".format(k, tag): d, "se_{}{}".format(k, tag): se, "w_{}{}".format(k, tag): "{}/{}".format(w, n)})
    if "met" in S and "met" in ref:
        res["d_met" + tag] = float(np.mean(list(S["met"].values()))) - float(np.mean(list(ref["met"].values())))


def evaluate(args):
    name, members, ref, ref_mbr = args
    t0 = time.time()
    doms = [d for d in ("dfl", "skc", "met", "pff") if has(d, members)]
    S = score_mbr(members, doms)
    res = dict(name=name, decoder="mbr", mbr_cfg=CFG, members=members, alone_dfl=None, domains=doms)
    if "dfl" in S:
        res["dfl"] = float(np.mean([r["fine4"] for r in S["dfl"].values()]))
        res["d_sel2"] = vo.paired_dfl({u: {"raw": r["raw"]} for u, r in S["dfl"].items()},
                                       {u: {"raw": r["raw"]} for u, r in ref["dfl"].items()}, "raw")[0]
    if "skc" in S:
        res["skc"] = float(np.mean([r["fine3"] for r in S["skc"].values()]))
    if "met" in S:
        res["met"] = float(np.mean(list(S["met"].values())))
    if "pff" in S:
        res["pff"] = float(np.mean([r["fine3"] for r in S["pff"].values()]))
    paired_vs(res, S, ref, "")              # vs v6 (hard vote, v5c rule): the prereg reference
    if ref_mbr is not None:
        paired_vs(res, S, ref_mbr, "_vs_mbrlsw")
    for k, (n, ng, mins) in S["poss"].items():
        res["p90_" + k] = round(90.0 * n / mins, 1)
        res["p90ref_" + k] = round(90.0 * ref["poss"][k][0] / ref["poss"][k][2], 1) if k in ref["poss"] else None
        res["p90gt_" + k] = round(90.0 * ng / mins, 1)
    if all(k in res for k in ("d_dfl", "d_skc", "d_met")):
        res["eligible"] = bool(res["d_dfl"] >= 0.005 and res["d_dfl"] >= 2 * res["se_dfl"]
                               and res["d_skc"] >= -res["se_skc"] and res["d_met"] >= -0.010)
    else:
        res["eligible"] = None
    res["secs"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--log", default=os.path.join(vr.WORK, "arch", "round4", "votes.jsonl"))
    ap.add_argument("--no-mbrlsw", action="store_true")
    a = ap.parse_args()
    vo.dfl_info(), vo.pff_info(), vo.skc_info(), vo.met_info()
    ref = vr.score_set(vr.REF, vr.RULE)
    lsw = ["o_ls05", "m5c_W", "radar_pff"]
    ref_mbr = None if a.no_mbrlsw else score_mbr(lsw, [d for d in ("dfl", "skc", "met", "pff") if has(d, lsw)])
    jobs = []
    for s in a.set:
        name, ms = s.split("=", 1)
        jobs.append((name, ms.split(","), ref, ref_mbr))
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for r in pool.imap(evaluate, jobs):
            print(vr.line(r) + " | vs MBR-lsw DFL {} SKC {} Met {}".format(
                "{:+.4f}±{:.4f}".format(r["d_dfl_vs_mbrlsw"], r["se_dfl_vs_mbrlsw"]) if "d_dfl_vs_mbrlsw" in r else "-",
                "{:+.4f}±{:.4f}".format(r["d_skc_vs_mbrlsw"], r["se_skc_vs_mbrlsw"]) if "d_skc_vs_mbrlsw" in r else "-",
                "{:+.4f}".format(r["d_met_vs_mbrlsw"]) if "d_met_vs_mbrlsw" in r else "-"), flush=True)
            with open(a.log, "a") as h:
                h.write(json.dumps(dict(r, time=time.strftime("%Y-%m-%d %H:%M")), default=float) + "\n")


if __name__ == "__main__":
    main()
