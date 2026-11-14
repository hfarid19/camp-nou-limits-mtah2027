"""v12 member copy (candidates/newfeat_v12): the newfeat experiment's module (arch/r5_arch/newfeat/code), unchanged except
that nf_features is the sibling file loaded by path. newfeat INT(a) group (EXPERIMENT ONLY): 13 hand-picked interaction columns between ball kinematics (base + ACC) and
proximity / team columns (prereg addendum 2026-11-13 INT). Uses only base / ACC quantities that do not depend on the attack
direction, so fe.compute is called with sign = +1 (asserted equal for these columns in unit_chk_int.py).

inter(players, slot_team, ball_raw, pitch, fps, frames=None) -> E (len(frames) or N, 13) float32, names, kinds
Inputs (NaN -> 0 before the product; every column is 0 where the ball is missing, = the ACC mask):
  g_T = 1 / (1 + T_d1) (T's nearest-player distance; 0 if T has no tracked player), acc = ACC ball_acc (signed m/s^2),
  turn = ACC ball_turn, spd = base ball_speed, seen = base ball_seen (raw ball observed), dd = base diff_d1 clipped to +-20,
  T_close = base H_close / A_close (player-velocity closing speed of T's nearest player), lastctrl = base lastctrl (+-1 / 0),
  T_c2 = base H_c2 / A_c2 (players within 2 m), accmx = ACC ball_acc_mx0.4.
Columns (kinds): H_int_acc_prox = |acc| g_H, A_int_acc_prox (pair); H_int_turn_prox = turn g_H, A_int_turn_prox (pair);
  int_spd_oppclose = spd x close of the nearest player of the team NOT owning the overall nearest player (tie: mean) (sym);
  int_dd_acc = dd x |acc| (anti); int_dd_turn = dd x turn (anti); int_seen_spd = seen x spd, int_seen_acc = seen x |acc|,
  int_seen_turn = seen x turn (sym); int_lastctrl_turn = lastctrl x turn (anti); H_int_c2_accmx = H_c2 x accmx,
  A_int_c2_accmx (pair).
"""

import numpy as np

import importlib.util  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_NFN = "mtah_newfeat_v12_nf_features"
if _NFN not in sys.modules:                     # the sibling nf_features.py, loaded once per process under a private name
    _s = importlib.util.spec_from_file_location(_NFN, os.path.join(_HERE, "nf_features.py"))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_NFN] = _m
    _s.loader.exec_module(_m)
nf = sys.modules[_NFN]

fe = nf.fe


def inter(players, slot_team, ball_raw, pitch, fps, frames=None):
    n = len(ball_raw)
    fr = np.arange(n) if frames is None else np.asarray(frames, np.int64)
    F, names, _ = fe.compute(players, slot_team, ball_raw, np.ones(n), pitch, fps)
    A, an, _ = nf.extra(players, slot_team, ball_raw, pitch, fps, ("acc",))
    col = {k: F[:, i].astype(np.float64) for i, k in enumerate(names)}
    col.update({k: A[:, i].astype(np.float64) for i, k in enumerate(an)})
    _, seen = nf._ball_filled(ball_raw, fps)
    z = lambda x: np.nan_to_num(x, nan=0.0)  # noqa: E731
    gH = np.where(np.isnan(col["H_d1"]), 0.0, 1.0 / (1.0 + z(col["H_d1"])))
    gA = np.where(np.isnan(col["A_d1"]), 0.0, 1.0 / (1.0 + z(col["A_d1"])))
    acc, turn, spd = np.abs(col["ball_acc"]), col["ball_turn"], z(col["ball_speed"])
    dd = np.clip(z(col["diff_d1"]), -20, 20)
    dH = np.where(np.isnan(col["H_d1"]), np.inf, col["H_d1"])
    dA = np.where(np.isnan(col["A_d1"]), np.inf, col["A_d1"])
    cH, cA = z(col["H_close"]), z(col["A_close"])
    opp = np.where(dH < dA, cA, np.where(dA < dH, cH, 0.5 * (cH + cA)))
    cols, out_n, out_k = [], [], []

    def add(name, v, kind="sym"):
        v = np.where(seen, np.nan_to_num(np.asarray(v, np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        cols.append(v[fr].astype(np.float32))
        out_n.append(name)
        out_k.append(kind)

    add("H_int_acc_prox", acc * gH, "H:int_acc_prox")
    add("A_int_acc_prox", acc * gA, "A:int_acc_prox")
    add("H_int_turn_prox", turn * gH, "H:int_turn_prox")
    add("A_int_turn_prox", turn * gA, "A:int_turn_prox")
    add("int_spd_oppclose", spd * opp)
    add("int_dd_acc", dd * acc, "anti")
    add("int_dd_turn", dd * turn, "anti")
    add("int_seen_spd", col["ball_seen"] * spd)
    add("int_seen_acc", col["ball_seen"] * acc)
    add("int_seen_turn", col["ball_seen"] * turn)
    add("int_lastctrl_turn", col["lastctrl"] * turn, "anti")
    add("H_int_c2_accmx", z(col["H_c2"]) * col["ball_acc_mx0.4"], "H:int_c2_accmx")
    add("A_int_c2_accmx", z(col["A_c2"]) * col["ball_acc_mx0.4"], "A:int_c2_accmx")
    return np.stack(cols, 1).astype(np.float32), out_n, out_k
