#!/usr/bin/env python3
"""Parity of the numpy pointer forward (ptr_np) with the torch model (ptr_net), on a stretch of one ds file.

    python -I candidates/nn_pointer/verify_ptr.py <model.npz> <ds file .npz> [T]
Prints max |diff| of the team probabilities, hand-over probability and pointer softmax (must be <= 1e-4).
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptr_net as PN  # noqa: E402
import ptr_np         # noqa: E402


def main():
    mp, dp = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    W, meta = ptr_np.load(mp)
    base = os.path.dirname(os.path.dirname(os.path.abspath(dp)))
    z = np.load(dp)
    pz = np.load(os.path.join(base, "ptrb" if meta["cfg"].get("ballc", 0) else "ptr", os.path.basename(dp)))
    s0 = len(z["idx"]) // 3
    sl = slice(s0, s0 + T)
    Xn = np.nan_to_num((z["X"][sl].astype(np.float64) - meta["mean"]) / meta["std"])
    xs, ms, xf = (pz[k][sl].astype(np.float64) for k in ("xs", "ms", "xf"))
    used = ms.max(0) > 0
    xs, ms = xs[:, used], ms[:, used]
    o = ptr_np.forward(W, meta, xs, ms, xf, Xn)
    cfg = meta["cfg"]
    model = PN.PointerNet(len(meta["names"]), cfg).double()
    zz = np.load(mp)
    model.load_state_dict({k[2:]: torch.from_numpy(zz[k].astype(np.float64)) for k in zz.files if k.startswith("w/")})
    model.eval()
    with torch.no_grad():
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]  # noqa: E731
        r = model(t(xs), t(ms), t(xf), t(Xn.T))
        lt, lsm = PN.team_probs(r["logits"], r["team"], t(ms))
        P = lt.exp()[0].numpy()
        if cfg.get("mix", 0) > 0:
            P = (1 - cfg["mix"]) * P + cfg["mix"] * torch.softmax(r["dir"], 1)[0].T.numpy()
        ho = torch.sigmoid(r["ho"])[0].numpy()
        g = lsm.exp()[0].numpy()
    d = dict(P=float(np.abs(P - o["P"]).max()), ho=float(np.abs(ho - o["ho"]).max()), g=float(np.abs(g - o["g"]).max()))
    print(json.dumps(d), "T", T, "S", int(used.sum()), "OK" if max(d.values()) <= 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
