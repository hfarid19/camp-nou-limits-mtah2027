"""Shared pieces of the R5-4 auxiliary-target trainers: the aux head, the team swap of the targets, the loss.

The aux head reads the main network's last hidden layer and is trained jointly; it is NOT exported
(the exported npz holds exactly the main network's weights, so the members' numpy inference is unchanged).
Groups (aux_dataset make_aux.py): ball (0-3: x, |y|, vx, speed of the clean ball; frames whose input
ball is missing are up-weighted by 1 + aux_miss), time (4-7: log time to / since the next / last
H<->A and alive<->dead change), restart (team of the next possession on dead frames; BCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_CFG = dict(aux_lam=0.3, aux_ball=1.0, aux_time=1.0, aux_rst=1.0, aux_miss=1.0, aux_hidden=64, aux_beta=0.05)


class AuxHead(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(C, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, 9, 1))

    def forward(self, h):
        return self.net(h)


def swap_aux(a):
    """team swap of one crop's aux (10, T) tensor: anti channels 0, 2 negated; restart team 0 <-> 1."""
    a = a.clone()
    a[0] = -a[0]
    a[2] = -a[2]
    r = a[8]
    a[8] = torch.where(r >= 0, 1 - r, r)
    return a


def aux_loss(pred, aux, valid, cfg):
    """pred (B, 9, T), aux (B, 10, T), valid (B, T) bool -> (total, dict of parts)."""
    tgt = aux[:, :8]
    m = (~torch.isnan(tgt)) & valid[:, None]
    l = F.smooth_l1_loss(pred[:, :8], torch.nan_to_num(tgt), beta=cfg["aux_beta"], reduction="none")
    wf = 1.0 + cfg["aux_miss"] * aux[:, 9:10]
    mf = m.float()
    lb = (l[:, :4] * mf[:, :4] * wf).sum() / ((mf[:, :4] * wf).sum() + 1)
    lt = (l[:, 4:] * mf[:, 4:]).sum() / (mf[:, 4:].sum() + 1)
    r = aux[:, 8]
    mr = ((r >= 0) & valid).float()
    lr = (F.binary_cross_entropy_with_logits(pred[:, 8], r.clamp(min=0), reduction="none") * mr).sum() / (mr.sum() + 1)
    tot = cfg["aux_lam"] * (cfg["aux_ball"] * lb + cfg["aux_time"] * lt + cfg["aux_rst"] * lr)
    return tot, dict(ball=float(lb.detach()), time=float(lt.detach()), rst=float(lr.detach()))
