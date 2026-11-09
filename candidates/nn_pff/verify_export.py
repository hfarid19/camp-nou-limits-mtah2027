#!/usr/bin/env python3
"""Check that an exported model (trained on MPS or CPU) gives the same outputs in the numpy
inference code as in torch on CPU and on MPS (float32), on a real training sequence.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -I candidates/nn_pff/verify_export.py mstcn|tcn MODEL.npz
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
kind, path = sys.argv[1], sys.argv[2]
WORK = os.environ["MTAH_WORK"]
z = np.load(os.path.join(WORK, "ds_pff", "ds", "PFF-10504_41.npz"))
X = z["X"][:3000].astype(np.float64)
W = np.load(path)
if kind == "mstcn":
    sys.path.insert(0, HERE)
    import tcn_np
    import train_mstcn as tm
    m = tcn_np.Single(W)
    Xn = m.normalise(X)
    pc_np, pb_np = m.forward(Xn)
    cfg = dict(tm.CFG, **json.loads(str(W["cfg"])))
    net = tm.ASRF(X.shape[1], cfg)
    net.load_state_dict({k[2:]: torch.from_numpy(W[k]) for k in W.files if k.startswith("w/")})
    net.eval()
    outs = {}
    for dev in ("cpu", "mps"):
        n2 = net.to(dev)
        with torch.no_grad():
            cs, bs = n2(torch.from_numpy(Xn.T[None].astype(np.float32)).to(dev))
        outs[dev] = (torch.softmax(cs[-1], 1)[0].T.cpu().numpy(), torch.sigmoid(bs[-1])[0].T.cpu().numpy())
    for dev, (pc, pb) in outs.items():
        print("{} vs numpy: class max |diff| {:.2e}, boundary {:.2e}".format(dev, np.abs(pc - pc_np).max(), np.abs(pb - pb_np).max()))
else:
    sys.path.insert(0, os.path.join(CAND, "nn_pff_tcn"))
    import tcn_np as tn
    sys.path.insert(0, HERE)
    import train_tcn as tt
    Wd = {k: W[k] for k in W.files}
    zin = tn.prep(X.astype(np.float32), Wd["mu"], Wd["sd"], Wd["ind"])
    lo_np = tn.forward(Wd, zin)
    net = tt.TCN(zin.shape[1], int(W["inp.weight"].shape[0]), [int(d) for d in W["dils"]])
    net.load_state_dict({k: torch.from_numpy(W[k]) for k in net.state_dict()})
    net.eval()
    for dev in ("cpu", "mps"):
        with torch.no_grad():
            lo = net.to(dev)(torch.from_numpy(zin.T[None].copy()).to(dev))[0].T.cpu().numpy()
        print("{} vs numpy: logits max |diff| {:.2e}, softmax {:.2e}".format(
            dev, np.abs(lo - lo_np).max(), np.abs(tn.softmax3(lo[:, :3]) - tn.softmax3(lo_np[:, :3])).max()))
