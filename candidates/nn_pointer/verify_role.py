#!/usr/bin/env python3
"""Parity of the numpy forward (ptr_np) with the torch model (ptr_net) for r5_ptr_role, as verify_ptr.py but on the
role inputs (<ds>/../ptrr), plus the team-swapped inputs (role.swap):

    python -I candidates/nn_pointer/verify_role.py <model.npz> <ds file .npz> [T]
Prints max |diff| of the team probabilities, hand-over probability and pointer softmax (must be <= 1e-4).
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ball_radar_pff"))
sys.path.insert(0, HERE)
import ptr_net as PN  # noqa: E402
import ptr_np         # noqa: E402
import role           # noqa: E402


def main():
    mp, dp = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    W, meta = ptr_np.load(mp)
    cfg = meta["cfg"]
    assert cfg.get("role", 0) == 1, cfg
    base = os.path.dirname(os.path.dirname(os.path.abspath(dp)))
    z = np.load(dp)
    pz = np.load(os.path.join(base, "ptrr", os.path.basename(dp)))
    s0 = len(z["idx"]) // 3
    sl = slice(s0, s0 + T)
    Xn = np.nan_to_num((z["X"][sl].astype(np.float64) - meta["mean"]) / meta["std"])
    xs, ms, xf = (pz[k][sl].astype(np.float64) for k in ("xs", "ms", "xf"))
    model = PN.PointerNet(len(meta["names"]), cfg).double()
    zz = np.load(mp)
    model.load_state_dict({k[2:]: torch.from_numpy(zz[k].astype(np.float64)) for k in zz.files if k.startswith("w/")})
    model.eval()
    res = {}
    for tag, (a, b) in (("", (xs, xf)), ("swap_", role.swap(xs, xf))):
        o = ptr_np.forward(W, meta, a, ms, b, Xn)
        with torch.no_grad():
            t = lambda v: torch.from_numpy(np.ascontiguousarray(v))[None]  # noqa: E731
            r = model(t(a), t(ms), t(b), t(Xn.T))
            lt, lsm = PN.team_probs(r["logits"], r["team"], t(ms))
            P = lt.exp()[0].numpy()
            ho = torch.sigmoid(r["ho"])[0].numpy()
            g = lsm.exp()[0].numpy()
        res.update({tag + "P": float(np.abs(P - o["P"]).max()), tag + "ho": float(np.abs(ho - o["ho"]).max()),
                    tag + "g": float(np.abs(g - o["g"]).max())})
    print(json.dumps(res), "T", T, "OK" if max(res.values()) <= 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
