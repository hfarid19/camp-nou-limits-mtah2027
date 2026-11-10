"""Bench units with the members' saved probabilities and codes, and the benches' scores.

Domains (unit keys):
  dfl  (match, seed)  21 game-seeds, LOGO member outputs, probs/nn_pff_dfl/<m>/dfl2_<match>_<seed>[_codes].npy
                      targets raw / T0.5 / T1 / evt1 -> fine4 (score_sel2.FINE)
  pff  game           48 PFF games, out-of-fold, probs/nn_pff/<m>/pff2_<g>_2027[_codes].npy; fine3 (screen only)
  skc  game           20 SkillCorner games, full models, probs/skc_mbr/<m>/skc2_<g>_2027[_codes].npy; fine3
  met  (game, seed)   Metrica dev (Sample Games 1, 2 x 2 seeds), probs/nn_pff_metrica/<m>/dev_<g>_<seed>[_codes].npy;
                      practice labels (spellF1)
Targets and blocks come from candidates/nn_pff/vote_offline.py (its cached pickles).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
import multiprocessing  # noqa: E402,F401  (stdlib queue before candidates/nn_pff/queue.py can shadow it)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("vote_offline", os.path.join(CAND, "nn_pff", "vote_offline.py"))
vo = importlib.util.module_from_spec(_spec)              # by path: candidates/nn_pff/queue.py shadows stdlib queue
sys.modules["vote_offline"] = vo
_spec.loader.exec_module(vo)
import possession_metrics as pm  # noqa: E402

WORK = vo.WORK
PROBS = os.path.join(WORK, "probs")
V6 = ("t5c", "m5c", "radar_pff")
LSW = ("o_ls05", "m5c_W", "radar_pff")


def units(domain):
    if domain == "dfl":
        return sorted(vo.dfl_info())
    if domain == "pff":
        return sorted(vo.pff_info())
    if domain == "skc":
        return sorted(vo.skc_info())
    if domain == "met":
        return sorted(_met())
    raise ValueError(domain)


_MET = {}


def _met():
    if not _MET:
        _MET.update(vo.met_info())
    return _MET


def info(domain, u):
    """-> (blocks, targets dict)."""
    if domain == "dfl":
        b, tg = vo.dfl_info()[u]
        return b, {k: tg[k] for k in ("raw", "T0.5", "T1", "evt1")}
    if domain == "pff":
        b, tg = vo.pff_info()[u]
        return b, {k: tg[k] for k in ("raw", "T0.5", "T1")}
    if domain == "skc":
        return vo.skc_info()[u]
    if domain == "met":
        b, gt = _met()[u]
        return b, {"raw": gt}


def _path(domain, m, u):
    if domain == "dfl":
        return os.path.join(PROBS, "nn_pff_dfl", m, "dfl2_{}_{}".format(*u))
    if domain == "pff":
        return os.path.join(PROBS, "nn_pff", m, "pff2_{}_2027".format(u))
    if domain == "skc":
        return os.path.join(PROBS, "skc_mbr", m, "skc2_{}_2027".format(u))
    if domain == "met":
        return os.path.join(PROBS, "nn_pff_metrica", m, "dev_{}_{}".format(*u))


def member(domain, m, u):
    """-> (probs (N, 3) float64, codes int8)."""
    f = _path(domain, m, u)
    return np.load(f + ".npy").astype(np.float64), np.load(f + "_codes.npy").astype(np.int8)


def f1(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(np.asarray(pred, np.int8), runs), runs)["mean_f1"]


def score(domain, pred, targets):
    r = {k: f1(g, pred) for k, g in targets.items()}
    if domain == "dfl":
        r["main"] = float(np.mean([r[k] for k in ("raw", "T0.5", "T1", "evt1")]))
    elif domain in ("pff", "skc"):
        r["main"] = float(np.mean([r[k] for k in ("raw", "T0.5", "T1")]))
    else:
        r["main"] = r["raw"]
    return r


def paired(a, b, domain):
    """a, b: {unit: main score}. DFL: SE over the 7 game means; others: over units (Metrica: 2 games x 2
    seeds, SE over the 2 game means)."""
    common = sorted(set(a) & set(b))
    if domain in ("dfl", "met"):
        games = sorted({u[0] for u in common})
        d = np.array([np.mean([a[u] - b[u] for u in common if u[0] == g]) for g in games])
    else:
        d = np.array([a[u] - b[u] for u in common])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    wins = sum(a[u] > b[u] for u in common)
    return float(d.mean()), float(se), int(wins), len(common)
